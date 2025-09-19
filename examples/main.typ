#set page(height: auto,width: auto)
#import "layout.typ": layout
#import "@preview/fletcher:0.5.8" as fletcher: cetz
#import "edge_style.typ":gluon,top,photon,topp,tops,quark
#show raw: it => [
  #{if it.at("lang") == "dot"{
    layout(it.at("text"),scope:(gluon:gluon,tops:tops,top:top,topp:topp,photon:photon))
  }else{
    it
  }
}
]

```dot
 digraph dot_80_0_GL208 {
    spring_constant = 35;
    spring_length = 0.14;
    edge_vertex_repulsion = 1.355;
    charge_constant_v = 3.9;
    charge_constant_e = 0.7;
    delta=0.5;
    n_iters = 10001;
    temp = 0.367;

    node[
      eval="(stroke:blue,fill :black,
      radius:2pt,
      outset: -2pt)"
    ]

    edge[
      eval=top
    ]
    v0[pin="2.2,1.2", style=invis]
    v1[pin="2.2,-1.2",style=invis]
    v2[pin="-2.2,1.2",style=invis]
    v3[pin="-2.2,-1.2",style=invis]
    v0 -> v11 [eval=photon]
    v1 -> v10 [eval="(..photon,label:[$gamma$],label-side: left)", mom_eval="(label:[$p_1$],label-sep:0mm)"]
    v9 -> v2 [eval=photon]
    v8 -> v3 [eval=photon]
    v4 -> v10
    v10 -> v5
    v5 -> v11
    v11 -> v4
    v4 -> v7 [eval=gluon]
    v5 -> v6 [eval=gluon]
    v6 -> v8
    v8 -> v7
    v7 -> v9
    v9 -> v6
}
```
