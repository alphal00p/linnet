// Default per-graph Typst template bundled with the linnet CLI.
// The CLI passes --input title="..." and --input data="..." for every build.
#import "layout.typ": layout
#import "@preview/fletcher:0.5.8" as fletcher: cetz
#show raw: it => [
  #{if it.at("lang") == "dot"{
    layout(it.at("text"))
  }else{
    it
  }
}
]

#set page(width: 180mm, height: auto, margin: (x: 14mm, y: 16mm))

#let title = sys.inputs.at("title", default: "A")
#let data_path = sys.inputs.at("data_path", default: none)

#align(center)[
  #text(size: 18pt, weight: "semibold")[#title]
]

#box(
  inset: 14pt,
  radius: 6pt,
  stroke: 1pt + gray,
  width: 100%,
)[
  #if data_path == none {
    text(fill: gray)[No data path provided.]
  } else {
    let text = read(data_path)
    raw(text, lang: "dot")
  }
  // #text(fill: gray)[Placeholder figure for #data_path]
  // TODO: import and visualize the DOT data here.
]
