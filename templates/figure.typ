// Default per-graph Typst template bundled with the linnet CLI.
// The CLI passes --input title="..." and --input data="..." for every build.

#set page(width: 180mm, height: auto, margin: (x: 14mm, y: 16mm))

#let title = sys.inputs.at("title", default: "A")
#let data_path = sys.inputs.at("data", default: none)

#align(center)[
  #text(size: 18pt, weight: "semibold")[#title]
]

#v(6pt)
#text(size: 10pt, fill: gray)[Source: #data_path]

#v(12pt)
#box(
  inset: 14pt,
  radius: 6pt,
  stroke: 1pt + gray,
  width: 100%,
)[
  #text(fill: gray)[Placeholder figure for #data_path]
  // TODO: import and visualize the DOT data here.
]
