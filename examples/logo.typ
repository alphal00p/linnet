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
 digraph linnet {
    // spring_constant = 40;
    // spring_length = 0.14;
    // edge_vertex_repulsion = 0.355;
    // charge_constant_v = 10;
    // charge_constant_e = 0.7;
    // delta=1.42;

    spring_constant = 55;
    spring_length = 0.14;
    edge_vertex_repulsion = 1.355;
    charge_constant_v = 79;
    charge_constant_e = 0.1;
    delta=.91;
    n_iters = 30001;
    seed=42
    temp = .6;

    node[
      eval="(stroke:red.lighten(40%)+2mm,fill :black,
      radius:3mm,
      outset: -2pt)"
    ]

    edge[
      eval="(stroke:gray.lighten(0%)+3.8mm)"
    ]
    a1->c
    a2->c
    a3->c
    a4->c
    a5->c
    a6->c
    // ll->c
    a7->c
    c [pin="0.,0."]
    a2 [pin="-2,-1"]
    a7->l

    l [pin="2.,3.", eval="(radius:1mm)"]
    le [style=invis]
    l->le
    l->le
    l->le

    a3->b3
    a4->b4
    b3->b4

    a5->b4

   

    b4->p 
    b3->p


    // ext [style=invis]

    // ext -> pp
    // ext -> p
    // ext -> b3
    b4->pp
    b4->ppp
    eye [pin="1000,0",shift="-998.5,-4.3",eval="(stroke:blue+2mm,fill :black,
       radius:4mm,extrude:(0,1,),
       outset: -2pt)"]
    p->pp
    pp->ppp
    ppp [pin="4,-3.5" shift="-1.2,-.5"]
    pp  [shift="-1,0" ]
    a1->a2->a3->a4->a5->a6->a7->a1

    a1->t
    t->a2

    a1->t1
    t1->t
    // t2->t1
    // t2->tt
    t->tt

    tt->t1
    tt->ttt
    // t2->ttt
    tttt->ttt

    tttt [style=invis, pin="-7,5"]
}
```
