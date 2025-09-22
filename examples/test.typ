#set page(height: auto,width: auto)
#import "layout.typ": layout 
#import "@preview/fletcher:0.5.8" as fletcher: cetz
#import "edge_style.typ":gluon,top,photon,topp,tops,quark
#show raw: it => [
  #{if it.at("lang") == "dot"{
    layout(it.at("text"),scope:(gluon:gluon,tops:tops,top:top,topp:topp,dx:quark,g:gluon,d:quark,photon:photon,a:photon))
  }else{
    it
  }
}
]

```dot
digraph{
  a->b
}

```