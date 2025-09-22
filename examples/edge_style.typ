#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, cetz,edge,hide
#let top =(stroke:black+0.5mm,marks:((inherit:"solid",rev:false,pos:0.5,scale:50%),))

#let down =(stroke:red.lighten(60%)+0.5mm,marks:((inherit:"solid",rev:false,pos:0.5,scale:50%),))
// #let down =(stroke:(paint: black, thickness: 0.5mm, dash: "dotted"),marks:((inherit:"solid",rev:false,pos:0.5,scale:50%),))

#let tops =(stroke:black+0.5mm,marks:((inherit:"solid",rev:false,pos:0.4,scale:50%),))
#let topp =(stroke:black+0.5mm,marks:((inherit:"solid",rev:false,pos:0.6,scale:50%),))
 // marks:((inherit:"|>",pos:0.5,scale:60%),)
#let photon =(stroke:(paint:black),decorations:
cetz.decorations.wave
.with(amplitude: 3pt,segment-length:0.2))
#let quark =(stroke:black+0.5mm)
#let gluon =(stroke:(paint:black)
,decorations:
cetz.decorations.coil
.with(amplitude: 5pt,segment-length:0.145 , align:"END"))
#let mom_arr =(stroke:black+1.1pt,marks:((inherit:"head",rev:false,pos:1,scale:40%),))
#let initial = (thickness:0.5mm,paint:rgb(100%,0%,0%),cap:"round")
#let final = (thickness:0.5mm,paint:rgb(0%,0%,100%),cap:"round")
       
