#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, cetz,edge,hide
#let mom_arr =(stroke:black+0.3mm,marks:((inherit:"head",rev:false,pos:1,scale:40%),))
#let p = plugin("./linnest.wasm")


#let layout(input,scope:(:))={
  let a= p.layout_graph(bytes(input),cbor.encode((:)));


  let graphs= cbor(a);

  let diags = ();
  for (g,parse) in graphs{

  let noed = ();
  let n =(:);

  for (i,v) in g.nodes.enumerate(){
    n.insert(str(i),v)
    let (x,y) = v.remove("pos")
    let b= v.remove("shift")
    let ev = v.remove("eval",default:"(:)")
    if ev == none{
      ev = "(:)"
    }
    noed.push(node(pos:(x,y),name:label(str(i)),..eval(ev,scope: scope ,mode: "code"),layer:2))
    // v.percent=xpercent
  }


  for (i,e) in g.edges.enumerate(){
    let start = e.data.remove("from")
    let end = e.data.remove("to")
    let data = e.remove("data")

    let ev = data.remove("eval",default:"(:)")
    if ev == none{
      ev = "(:)"
    }

    let mev = data.remove("mom_eval",default:"(:)")
    if mev == none{
      mev = "(:)"
    }
    let bend
    let o = e.remove("orientation")

    let snmlab = label("sm"+str(i))
    let (start-node,start-node-pos) =if start != none{
      let nodelab = label(str(start))

      (nodelab,n.at(str(start)).pos)
    } else{
      let lab = label("exts"+str(i))
      noed.push(node((data.pos.x,data.pos.y),name:lab,outset:-5mm,radius:5mm,fill:none))
      (lab,data.pos)
    }

    let enmlab = label("em"+str(i))
    let (end-node,end-node-pos) =if end != none{
      let nodelab = label(str(end))

      (nodelab,n.at(str(end)).pos)
    } else{
      let lab = label("exte"+str(i))
      noed.push(node((data.pos.x,data.pos.y),name:lab,outset:-5mm,radius:5mm,fill:none))
      (lab,data.pos)
    }


    let bend-angle = data.remove("bend")

    let bend = bend-angle.remove("Ok",default:0.)

    let percentb = 1+calc.abs(bend/calc.pi)

    let a  =  calc.sqrt(calc.pow(start-node-pos.x - end-node-pos.x,2)+calc.pow(start-node-pos.y - end-node-pos.y,2))*2.5mm*percentb

    noed.push(node(pos:(start-node-pos.x,start-node-pos.y),name:snmlab,outset:a))
    noed.push(node(pos:(end-node-pos.x,end-node-pos.y),name:enmlab,outset:a))


    let shift = if bend < 0. {
      1
    } else {
      -1
    } * 1.5mm

    let e = eval(ev,scope: scope ,mode: "code")

    if o == "Reversed"{

      noed.push(edge(vertices:(end-node,start-node),bend: bend * 1rad,..eval(ev,scope: scope ,mode: "code")))
    }else{

    noed.push(edge(vertices:(start-node,end-node),bend: bend * -1rad,..eval(ev,scope: scope ,mode: "code")))
    }


    noed.push(edge(vertices:(snmlab,enmlab),bend:bend * (percentb - 0.65) * -1rad,shift:shift,..mom_arr,..eval(mev,scope: scope ,mode: "code")))

  }


 diags.push( diagram(
 node-shape:circle,
	node-fill: black,
 edge-stroke:0.1em,
	spacing: 2em,
  ..noed,
))

// diags.push([#parse ])
  }

  grid(
    columns: 3,
    ..diags
  )
  }
