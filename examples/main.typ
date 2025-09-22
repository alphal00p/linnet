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

// Hard to get an intuition for the hamiltonian parameters ahah
// I wonder if there is not something to do with adding terms particular to the fundamental cycles of the graph

```dot
 digraph dot_80_0_GL208 {

    steps=600
    step=0.4
    beta =13.1
    k_spring=3.3;
    g_center=0
    gamma_dangling=50
    gamma_ee=0.3
    gamma_ev=0.01
    length_scale = 0.25
    node[
      eval="(stroke:blue,fill :black,
      radius:2pt,
      outset: -2pt)"
    ]

    edge[
      eval=top
    ]
    v0[pin="x:@initial,y:@p1", style=invis]
    v1[pin="x:@initial,y:@p2",style=invis]
    v2[pin="x:@final,y:@p1",style=invis]
    v3[pin="x:@final,y:@p2",style=invis]
    v0 -> v11 [eval=photon]
    v1 -> v10 [eval="(..photon,label:[$gamma$],label-side: left)", mom_eval="(label:[$p_1$],label-sep:0mm)"]
    v9 -> v2 [eval=photon]
    v8 -> v3 [eval=photon]
    v4 -> v10
    v10 -> v5
    v5 -> v11 [dir=back]
    v11 -> v4
    v4 -> v7 [eval=gluon]
    v5 -> v6 [eval=gluon]
    v6 -> v8
    v8 -> v7
    v7 -> v9
    v9 -> v6
}

//  digraph dot_80_0_GL208 {

//     steps=600
//     step=0.4
//     beta =13.1
//     k_spring=3.3;
//     g_center=0
//     gamma_dangling=50
//     gamma_ee=0.3
//     gamma_ev=0.01
//     length_scale = 0.25
//     node[
//       eval="(stroke:blue,fill :black,
//       radius:2pt,
//       outset: -2pt)"
//     ]

//     edge[
//       eval=top
//     ]
//     v0[pin="x:@initial,y:@p1", style=invis]
//     v1[pin="x:@initial,y:@p2",style=invis]
//     v2[pin="x:@final,y:@p1",style=invis]
//     v3[pin="x:@final,y:@p2",style=invis]
//     v0 -> v11 [eval=photon]
//     v1 -> v10 [eval="(..photon,label:[$gamma$],label-side: left)", mom_eval="(label:[$p_1$],label-sep:0mm)"]
//     v9 -> v2 [eval=photon]
//     v8 -> v3 [eval=photon]
//     v4 -> v10
//     v10 -> v5
//     v5 -> v11 [dir=back]
//     v11 -> v4
//     v4 -> v7 [eval=gluon]
//     v5 -> v6 [eval=gluon]
//     v6 -> v8
//     v8 -> v7
//     v7 -> v9
//     v9 -> v6
// }


digraph {


    steps=600
    step=0.4
    beta =13.1
    k_spring=3.3;
    g_center=0
    gamma_dangling=50
    gamma_ee=0.3
    gamma_ev=0.01
    length_scale = 0.25

    a->b
// // "v1" -> "v10" [label="p1 | a",color="blue",penwidth="0.6",style="solid"];
// // "v9" -> "v2" [label="p4 | a",color="blue",penwidth="0.6",style="solid"];
// // "v8" -> "v3" [label="p3 | a",color="blue",penwidth="0.6",style="solid"];
// // "v4" -> "v7" [label="q1 | g",color="red",penwidth="0.6",style="dashed"];
// // "v5" -> "v6" [label="q4 | g",color="red",penwidth="0.6",style="solid"];
// // "v6" -> "v7" [label="q7 | g",color="red",penwidth="0.6",style="dashed"];
// // "v6" -> "v7" [label="q8 | g",color="red",penwidth="0.6",style="solid"];
// // "v4" -> "v10" [label="q2 | t",color="black",penwidth="1.2",style="solid"];
// // "v10" -> "v8" [label="q9 | t",color="black",penwidth="1.2",style="dashed"];
// // "v8" -> "v5" [label="q5 | t",color="black",penwidth="1.2",style="solid"];
// // "v5" -> "v9" [label="q6 | t",color="black",penwidth="1.2",style="solid"];
// // "v9" -> "v11" [label="q10 | t",color="black",penwidth="1.2",style="solid"];
// // "v11" -> "v4" [label="q3 | t",color="black",penwidth="1.2",style="solid"];
}
```

// ```dot
// digraph qqx_aaa_pentagon {
//     steps=600
//     step=.2
//     beta =3.1
//     k_spring=15.3;
//     g_center=0
//     gamma_ee=0.3
//     gamma_ev=0.01
//     length_scale = 0.2

//      node[
//       eval="(stroke:blue,fill :black,
//       radius:2pt,
//       outset: -2pt)"
//     ]
//     // edge[
//     //   // eval="{particle}"
//     // ]

// // exte0 [style=invis pin="x:-4"];
// v3:0 -> exte0;
//  // exte1 [style=invis pin="x:4"];
// // exte1 -> vl1 [ particle="d"];
// // exte2 [style=invis pin="x:4"];
// // exte2 -> vl2:2 [particle="dx"];
// // exte3 [style=invis pin="x:-4"];
// // v1:3 -> exte3  [particle="a"];
// // exte4 [style=invis pin="x:-4"];
// // v2:4 -> exte4 [particle="a"];
// // v1:5 -> v2:6 [particle="d"];
// // v2:7 -> v3:8 [particle="d"];
// // vl1:11 -> v1:12 [particle="d"];
// // v3:9 -> vl2:10  [ particle="d"];
// // vl1:13 -> vl2:14 [particle="g"];
// }
// ```
