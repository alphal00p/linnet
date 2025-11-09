use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, ensure, Context, Result};
use blake3::Hasher;
use clap::{ArgAction, Parser};
use pathdiff::diff_paths;
use rust_embed::RustEmbed;
use serde_json::{self, Map as JsonMap, Value as JsonValue};
use walkdir::WalkDir;

const TEMPLATE_SUBDIR: &str = "templates";

#[derive(RustEmbed)]
#[folder = "templates"]
struct EmbeddedTemplates;

#[derive(Copy, Clone)]
enum TemplateKind {
    Figure,
    Grid,
}

impl TemplateKind {
    fn file_name(self) -> &'static str {
        match self {
            TemplateKind::Figure => "figure.typ",
            TemplateKind::Grid => "grid.typ",
        }
    }

    fn embedded_bytes(self) -> Result<Cow<'static, [u8]>> {
        EmbeddedTemplates::get(self.file_name())
            .map(|file| file.data)
            .ok_or_else(|| anyhow!("embedded template {} is missing", self.file_name()))
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        for cause in err.chain().skip(1) {
            eprintln!("  caused by: {cause}");
        }
        std::process::exit(1);
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "linnet",
    about = "Incrementally render Typst figures for every .dot file and assemble them into a grid PDF."
)]
struct Cli {
    /// Directory to scan for .dot files.
    root: PathBuf,
    /// Shared Typst template used for every individual figure.
    #[arg(
        long,
        value_name = "FILE",
        default_value = "build/templates/figure.typ"
    )]
    figure_template: PathBuf,
    /// Typst template used for the grid document.
    #[arg(long, value_name = "FILE", default_value = "build/templates/grid.typ")]
    grid_template: PathBuf,
    /// Base directory that stores build artifacts.
    #[arg(long, value_name = "DIR", default_value = "build")]
    build_dir: PathBuf,
    /// Directory for the generated figure PDFs (defaults to <build_dir>/figs).
    #[arg(long, value_name = "DIR")]
    figs_dir: Option<PathBuf>,
    /// Cache file storing hashes (defaults to <build_dir>/.cache/figures.json).
    #[arg(long, value_name = "FILE")]
    cache_file: Option<PathBuf>,
    /// Path for the generated fig-index.typ (defaults to the directory of grid_template).
    #[arg(long, value_name = "FILE")]
    fig_index: Option<PathBuf>,
    /// Destination PDF for the final grid (defaults to <build_dir>/grid.pdf).
    #[arg(long, value_name = "FILE")]
    grid_output: Option<PathBuf>,
    /// Extra files whose contents influence incremental rebuilds (e.g., style snippets).
    #[arg(long, value_name = "FILE", action = ArgAction::Append)]
    style: Vec<PathBuf>,
    /// Number of columns in the final grid.
    #[arg(long, value_name = "N", default_value_t = 3)]
    columns: usize,
}

#[derive(Clone)]
struct FigurePlan {
    data_path: PathBuf,
    relative: PathBuf,
    output_path: PathBuf,
    title: String,
}

#[derive(Clone)]
struct FigureRecord {
    output_path: PathBuf,
    relative: PathBuf,
    title: String,
}

#[derive(Default)]
struct FolderNode {
    figures: Vec<FigureEntry>,
    children: BTreeMap<String, FolderNode>,
}

struct FigureEntry {
    path: String,
    relative: String,
    title: String,
    name: String,
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.columns > 0, "columns must be greater than zero");

    let root = canonicalize_existing(&cli.root)
        .with_context(|| format!("failed to read root directory {}", cli.root.display()))?;
    let cwd = std::env::current_dir().context("failed to resolve working directory")?;
    let build_dir = absolutize(&cwd, &cli.build_dir);
    let figs_dir = cli
        .figs_dir
        .as_ref()
        .map(|path| absolutize(&cwd, path))
        .unwrap_or_else(|| build_dir.join("figs"));
    let cache_file = cli
        .cache_file
        .as_ref()
        .map(|path| absolutize(&cwd, path))
        .unwrap_or_else(|| build_dir.join(".cache").join("figures.json"));
    let grid_output = cli
        .grid_output
        .as_ref()
        .map(|path| absolutize(&cwd, path))
        .unwrap_or_else(|| build_dir.join("grid.pdf"));
    let requested_figure_template = absolutize(&cwd, &cli.figure_template);
    let requested_grid_template = absolutize(&cwd, &cli.grid_template);

    fs::create_dir_all(&build_dir)
        .with_context(|| format!("failed to create build directory {}", build_dir.display()))?;
    fs::create_dir_all(&figs_dir)
        .with_context(|| format!("failed to create figures directory {}", figs_dir.display()))?;
    if let Some(parent) = cache_file.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create cache directory {}", parent.display()))?;
    }

    let mut style_files = Vec::new();
    for style in &cli.style {
        let canonical = canonicalize_existing(style)
            .with_context(|| format!("failed to read style file {}", style.display()))?;
        style_files.push(canonical);
    }
    style_files.sort();
    style_files.dedup();

    let figure_template =
        resolve_template(&requested_figure_template, TemplateKind::Figure, &build_dir)?;
    let grid_template = resolve_template(&requested_grid_template, TemplateKind::Grid, &build_dir)?;
    let fig_index_path = cli
        .fig_index
        .as_ref()
        .map(|path| absolutize(&cwd, path))
        .unwrap_or_else(|| {
            grid_template
                .parent()
                .unwrap_or(Path::new("."))
                .join("fig-index.typ")
        });

    let dot_files = collect_dot_files(&root)?;
    if dot_files.is_empty() {
        println!("No .dot files found under {}", root.display());
        return Ok(());
    }

    let mut plans = Vec::new();
    for data_path in dot_files {
        let relative = diff_paths(&data_path, &root).ok_or_else(|| {
            anyhow!(
                "failed to compute relative path for {}",
                data_path.display()
            )
        })?;
        let mut output_path = figs_dir.join(&relative);
        output_path.set_extension("pdf");
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create output directory {} for {}",
                    parent.display(),
                    relative.display()
                )
            })?;
        }
        let title = derive_title(&relative);
        plans.push(FigurePlan {
            data_path,
            relative,
            output_path,
            title,
        });
    }

    plans.sort_by(|a, b| a.relative.cmp(&b.relative));

    let previous_cache = load_cache(&cache_file)?;
    let mut new_cache = BTreeMap::new();
    let mut records = Vec::new();
    let mut rebuilt = 0usize;
    let mut reused = 0usize;

    for plan in &plans {
        let key = path_key(&plan.relative);
        let hash = compute_hash(&plan.data_path, &figure_template, &style_files)?;
        if previous_cache
            .get(&key)
            .map(|old| old == &hash)
            .unwrap_or(false)
        {
            reused += 1;
        } else {
            build_figure(plan, &figure_template, &cwd)?;
            rebuilt += 1;
        }
        new_cache.insert(key, hash);
        records.push(FigureRecord {
            output_path: plan.output_path.clone(),
            relative: plan.relative.clone(),
            title: plan.title.clone(),
        });
    }

    save_cache(&cache_file, &new_cache)?;
    let removed = remove_stale_outputs(&previous_cache, &new_cache, &figs_dir)?;

    let grid_dir = grid_template
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    write_fig_index(&records, &fig_index_path, &grid_dir, cli.columns)?;
    compile_grid(&grid_template, &grid_output, &cwd)?;

    println!(
        "figures: {} built, {} reused{}",
        rebuilt,
        reused,
        if removed.is_empty() {
            String::new()
        } else {
            format!(", {} removed", removed.len())
        }
    );
    println!("grid: {}", grid_output.display());

    Ok(())
}

fn collect_dot_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(root).into_iter() {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .and_then(OsStr::to_str)
                .map(|ext| ext.eq_ignore_ascii_case("dot"))
                .unwrap_or(false)
        {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

fn canonicalize_existing(path: &Path) -> Result<PathBuf> {
    Ok(fs::canonicalize(path)?)
}

fn absolutize(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn derive_title(relative: &Path) -> String {
    let mut parts = Vec::new();
    for component in relative.components() {
        let part = component.as_os_str().to_string_lossy();
        parts.push(part);
    }
    let title = parts.join(" / ");
    if let Some(stripped) = title.strip_suffix(".dot") {
        stripped.to_owned()
    } else {
        title
    }
}

fn compute_hash(data: &Path, template: &Path, styles: &[PathBuf]) -> Result<String> {
    let mut hasher = Hasher::new();
    feed_file(&mut hasher, data)?;
    feed_file(&mut hasher, template)?;
    for style in styles {
        feed_file(&mut hasher, style)?;
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn feed_file(hasher: &mut Hasher, path: &Path) -> Result<()> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut buffer = [0u8; 8192];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {} while hashing", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(())
}

fn build_figure(plan: &FigurePlan, template: &Path, root: &Path) -> Result<()> {
    let mut command = Command::new("typst");
    command
        .arg("c")
        .arg(template)
        .arg(&plan.output_path)
        .arg("--root")
        .arg(root)
        .arg("--input")
        .arg(format!(
            "data=\"{}\"",
            escape_typst_string(&plan.data_path.to_string_lossy())
        ))
        .arg("--input")
        .arg(format!("title=\"{}\"", escape_typst_string(&plan.title)));

    run_typst(
        &mut command,
        &format!("building {}", plan.relative.display()),
    )
}

fn run_typst(command: &mut Command, description: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("failed to run typst while {description}"))?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "typst failed while {description}:\nstdout:\n{}\nstderr:\n{}",
            stdout.trim(),
            stderr.trim()
        );
    }
    Ok(())
}

fn escape_typst_string(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn path_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn load_cache(path: &Path) -> Result<HashMap<String, String>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read cache file {}", path.display()))?;
    if data.trim().is_empty() {
        return Ok(HashMap::new());
    }

    let parsed: JsonValue = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse cache file {}", path.display()))?;
    let mut map = HashMap::new();
    if let Some(figs) = parsed.get("figures").and_then(JsonValue::as_object) {
        for (key, value) in figs {
            if let Some(hash) = value.as_str() {
                map.insert(key.clone(), hash.to_owned());
            }
        }
    }
    Ok(map)
}

fn save_cache(path: &Path, cache: &BTreeMap<String, String>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create cache parent {}", parent.display()))?;
    }
    let mut figures = JsonMap::new();
    for (key, value) in cache {
        figures.insert(key.clone(), JsonValue::String(value.clone()));
    }
    let mut root = JsonMap::new();
    root.insert("figures".into(), JsonValue::Object(figures));
    let json = JsonValue::Object(root);
    let mut file = File::create(path)
        .with_context(|| format!("failed to open cache file {} for writing", path.display()))?;
    let serialized = serde_json::to_string_pretty(&json)?;
    file.write_all(serialized.as_bytes())
        .with_context(|| format!("failed to write cache file {}", path.display()))?;
    Ok(())
}

fn remove_stale_outputs(
    previous: &HashMap<String, String>,
    current: &BTreeMap<String, String>,
    figs_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    for key in previous.keys() {
        if current.contains_key(key) {
            continue;
        }
        let relative = Path::new(key);
        let mut pdf_path = figs_dir.join(relative);
        pdf_path.set_extension("pdf");
        match fs::remove_file(&pdf_path) {
            Ok(()) => removed.push(pdf_path),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("failed to delete stale figure {}", relative.display())
                });
            }
        }
    }
    Ok(removed)
}

fn write_fig_index(
    records: &[FigureRecord],
    index_path: &Path,
    grid_dir: &Path,
    columns: usize,
) -> Result<()> {
    ensure_parent_dir(index_path)?;
    let mut file = File::create(index_path)
        .with_context(|| format!("failed to create {}", index_path.display()))?;
    writeln!(file, "// Auto-generated by linnet. Do not edit manually.")?;
    writeln!(file, "#let cols = {}", columns)?;

    let mut entries = Vec::new();
    for record in records {
        let rel_path =
            diff_paths(&record.output_path, grid_dir).unwrap_or_else(|| record.output_path.clone());
        let rel_display = record.relative.to_string_lossy().replace('\\', "/");
        let file_stem = record
            .relative
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let folder_parts = folder_components(&record.relative);
        let entry = FigureEntry {
            path: rel_path.to_string_lossy().replace('\\', "/"),
            relative: rel_display,
            title: record.title.clone(),
            name: file_stem,
        };
        entries.push((folder_parts, entry));
    }

    let mut root = FolderNode::default();
    for (folders, entry) in entries {
        insert_entry(&mut root, &folders, entry);
    }

    write!(file, "#let tree = ")?;
    write_folder_node(&mut file, "root", &root, 0, false)?;
    writeln!(file)?;
    Ok(())
}

fn resolve_template(requested: &Path, kind: TemplateKind, build_dir: &Path) -> Result<PathBuf> {
    match fs::canonicalize(requested) {
        Ok(path) => return Ok(path),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(err)
                .with_context(|| format!("failed to read template {}", requested.display()));
        }
    }

    let mut target = requested.to_path_buf();
    if target.file_name().is_none() {
        target = target.join(kind.file_name());
    }

    let templates_dir = build_dir.join(TEMPLATE_SUBDIR);
    if !target.starts_with(&templates_dir) {
        bail!(
            "template {} not found and automatic creation is limited to {}",
            requested.display(),
            templates_dir.display()
        );
    }

    ensure_parent_dir(&target)?;
    let contents = kind.embedded_bytes()?;
    fs::write(&target, contents.as_ref())
        .with_context(|| format!("failed to write default template {}", target.display()))?;
    Ok(target)
}

fn folder_components(path: &Path) -> Vec<String> {
    path.parent()
        .map(|parent| {
            parent
                .components()
                .map(|component| component.as_os_str().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn typst_string_literal(value: &str) -> String {
    format!("\"{}\"", escape_typst_string(value))
}

fn format_typst_array(items: &[String]) -> String {
    if items.is_empty() {
        "()".to_string()
    } else {
        let body = items
            .iter()
            .map(|item| typst_string_literal(item))
            .collect::<Vec<_>>()
            .join(", ");
        format!("({body},)")
    }
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }
    Ok(())
}

fn compile_grid(template: &Path, output: &Path, root: &Path) -> Result<()> {
    ensure_parent_dir(output)?;
    let mut cmd = Command::new("typst");
    cmd.arg("c")
        .arg(template)
        .arg(output)
        .arg("--root")
        .arg(root);
    run_typst(&mut cmd, &format!("compiling grid {}", output.display()))
}

fn insert_entry(node: &mut FolderNode, folders: &[String], entry: FigureEntry) {
    if folders.is_empty() {
        node.figures.push(entry);
    } else {
        let (head, tail) = folders.split_first().unwrap();
        let child = node.children.entry(head.clone()).or_default();
        insert_entry(child, tail, entry);
    }
}

fn write_folder_node(
    file: &mut File,
    name: &str,
    node: &FolderNode,
    indent: usize,
    trailing_comma: bool,
) -> Result<()> {
    let indent_str = indent_spaces(indent);
    writeln!(file, "{indent_str}(")?;
    let field_indent = indent + 2;
    let field_str = indent_spaces(field_indent);
    writeln!(file, "{field_str}name: {},", typst_string_literal(name))?;
    write_figures_field(file, &node.figures, field_indent)?;
    let child_names: Vec<String> = node.children.keys().cloned().collect();
    writeln!(
        file,
        "{field_str}order: {},",
        format_typst_array(&child_names)
    )?;
    if child_names.is_empty() {
        writeln!(file, "{field_str}folders: (:),")?;
    } else {
        writeln!(file, "{field_str}folders: (")?;
        let key_indent = indent_spaces(field_indent + 2);
        for (child_name, child_node) in &node.children {
            write!(file, "{key_indent}{}: ", typst_string_literal(child_name))?;
            write_folder_node(file, child_name, child_node, field_indent + 4, true)?;
        }
        writeln!(file, "{field_str}),")?;
    }
    writeln!(
        file,
        "{indent_str}){}",
        if trailing_comma { "," } else { "" }
    )?;
    Ok(())
}

fn write_figures_field(file: &mut File, figures: &[FigureEntry], indent: usize) -> Result<()> {
    let indent_str = indent_spaces(indent);
    if figures.is_empty() {
        writeln!(file, "{indent_str}figures: (),")?
    } else {
        writeln!(file, "{indent_str}figures: (")?;
        let entry_indent = indent + 2;
        let entry_str = indent_spaces(entry_indent);
        for figure in figures {
            writeln!(
                file,
                "{entry_str}(path: {}, relative: {}, title: {}, name: {},),",
                typst_string_literal(&figure.path),
                typst_string_literal(&figure.relative),
                typst_string_literal(&figure.title),
                typst_string_literal(&figure.name)
            )?;
        }
        writeln!(file, "{indent_str}),")?;
    }
    Ok(())
}

fn indent_spaces(indent: usize) -> String {
    " ".repeat(indent)
}
