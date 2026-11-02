(function(){
"use strict";
const $=s=>document.querySelector(s);
const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ================= Views (explorer / guide) ================= */
let explorerStarted=false;
function route(){
  const g=location.hash==="#guide";
  $("#guide").hidden=!g;$("#explorer").hidden=g;
  if(g){window.scrollTo(0,0);}else if(!explorerStarted){explorerStarted=true;startExplorer();}
  else if(window.__azResize)window.__azResize();
}
window.addEventListener("hashchange",route);
route();

function startExplorer(){
if(typeof THREE==="undefined"||typeof Chess==="undefined"){const d=document.createElement("div");d.className="glerr";d.textContent="The 3D libraries didn't load. Check your connection and reload the page.";$("#stage").appendChild(d);return;}

/* ================= Game ================= */
const OPERA="e2e4 e7e5 g1f3 d7d6 d2d4 c8g4 d4e5 g4f3 d1f3 d6e5 f1c4 g8f6 f3b3 d8e7 b1c3 c7c6 c1g5 b7b5 c3b5 c6b5 c4b5 b8d7 e1c1 a8d8 d1d7 d8d7 h1d1 e7e6 b5d7 f6d7 b3b8 d7b8 d1d8".split(" ");
const G={fens:[],sans:[],moves:[],pos:[],reps:[],legal:[]};
function parseFEN(f){
  const [pl,side,cas,ep,half,full]=f.split(" ");
  const b=[...Array(8)].map(()=>Array(8).fill(null));
  pl.split("/").forEach((row,i)=>{let c=0;for(const ch of row){if(/\d/.test(ch))c+=+ch;else{b[7-i][c]=ch;c++;}}});
  return {b,side,cas,ep,half:+half,full:+full};
}
function rebuildDerived(from){
  G.pos.length=G.reps.length=G.legal.length=from;
  for(let i=from;i<G.fens.length;i++){
    G.pos[i]=parseFEN(G.fens[i]);
    const key=G.fens[i].split(" ").slice(0,4).join(" ");let n=0;for(let j=0;j<i;j++)if(G.fens[j].split(" ").slice(0,4).join(" ")===key)n++;G.reps[i]=n;
    const ch=new Chess(G.fens[i]);
    G.legal[i]={moves:ch.moves({verbose:true}),mate:ch.in_checkmate(),check:ch.in_check(),draw:ch.in_draw()||ch.in_stalemate()};
  }
}
function loadLine(ucis){
  const ch=new Chess();G.fens=[ch.fen()];G.sans=["Start"];G.moves=[null];
  for(const u of ucis){const m=ch.move({from:u.slice(0,2),to:u.slice(2,4),promotion:u[4]||"q"});if(!m)break;G.fens.push(ch.fen());G.sans.push(m.san);G.moves.push(m);}
  rebuildDerived(0);
}
function pushMove(m){
  G.fens.length=G.sans.length=G.moves.length=S.pos+1;
  const ch=new Chess(G.fens[S.pos]);const mv=ch.move(m);if(!mv)return false;
  G.fens.push(ch.fen());G.sans.push(mv.san);G.moves.push(mv);rebuildDerived(S.pos+1);return true;
}
const moveLabel=i=>{if(i===0)return "Start";const n=Math.ceil(i/2);return (i%2?n+". ":n+"… ")+G.sans[i];};
const PIECES="PNBRQKpnbrqk";
const PNAMES=["pawns","knights","bishops","rooks","queens","king"];
const FILES="abcdefgh";
const sqName=(r,c)=>FILES[c]+(r+1);
const sqRC=s=>({r:+s[1]-1,c:FILES.indexOf(s[0])});
const pieceName=ch=>({p:"pawn",n:"knight",b:"bishop",r:"rook",q:"queen",k:"king"})[ch.toLowerCase()];

/* ================= State ================= */
const S={pos:0,stage:"play",plane:0,variant:"B",cur:{r:4,c:1},revealed:64,sweeping:false,
  layer:2,chan:0,rf:{r:3,c:4},pol:56,from:{r:0,c:1},sel:null,playing:false,follow:false};

/* ================= Input tensor ================= */
function planeName(i){
  if(i<112){const t=Math.floor(i/14),j=i%14;const when=t===0?"Now":"T−"+t;
    if(j<12)return when+" · "+(j<6?"White ":"Black ")+PNAMES[j%6];
    return when+" · "+(j===12?"position seen before":"position seen twice before");}
  return ["Side to move (1 = White)","Total move count","White may castle kingside","White may castle queenside","Black may castle kingside","Black may castle queenside","Moves since capture or pawn move"][i-112];
}
function buildInput(p){
  const X=new Float32Array(119*64);
  for(let t=0;t<8;t++){const q=p-t;if(q<0)continue;const P=G.pos[q];
    for(let r=0;r<8;r++)for(let c=0;c<8;c++){const ch=P.b[r][c];if(ch)X[(t*14+PIECES.indexOf(ch))*64+r*8+c]=1;}
    if(G.reps[q]>=1)for(let s=0;s<64;s++)X[(t*14+12)*64+s]=1;
    if(G.reps[q]>=2)for(let s=0;s<64;s++)X[(t*14+13)*64+s]=1;}
  const P=G.pos[p];
  const cv=[P.side==="w"?1:0,P.full,P.cas.includes("K")?1:0,P.cas.includes("Q")?1:0,P.cas.includes("k")?1:0,P.cas.includes("q")?1:0,P.half];
  cv.forEach((v,k)=>{for(let s=0;s<64;s++)X[(112+k)*64+s]=v;});
  return X;
}

/* ================= Random-weight network (AlphaZero shapes, untrained) ================= */
function rng(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const R=rng(20171205);
const gauss=()=>{let u=0,v=0;while(!u)u=R();while(!v)v=R();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
function weights(co,ci){const W=new Float32Array(co*ci*9),s=Math.sqrt(2/(ci*9));for(let i=0;i<W.length;i++)W[i]=gauss()*s;return W;}
const W1=weights(256,119),TC=32;
const TW=[...Array(19)].map(()=>[weights(TC,TC),weights(TC,TC)]);
const VW=new Float32Array(TC).map(()=>gauss()/Math.sqrt(TC));
function conv(X,ci,W,co){
  const Y=new Float32Array(co*64);
  for(let i=0;i<ci;i++){const base=i*64;let any=false;for(let s=0;s<64;s++)if(X[base+s]!==0){any=true;break;}if(!any)continue;
    for(let o=0;o<co;o++){const wb=(o*ci+i)*9,yb=o*64;
      for(let r=0;r<8;r++)for(let c=0;c<8;c++){let acc=0;
        for(let dr=-1;dr<=1;dr++){const rr=r+dr;if(rr<0||rr>7)continue;
          for(let dc=-1;dc<=1;dc++){const cc=c+dc;if(cc<0||cc>7)continue;acc+=W[wb+(dr+1)*3+dc+1]*X[base+rr*8+cc];}}
        Y[yb+r*8+c]+=acc;}}}
  return Y;
}
function norm(Y,co){for(let o=0;o<co;o++){let m=0,v=0;for(let s=0;s<64;s++)m+=Y[o*64+s];m/=64;for(let s=0;s<64;s++){const d=Y[o*64+s]-m;v+=d*d;}const sd=Math.sqrt(v/64)+1e-5;for(let s=0;s<64;s++)Y[o*64+s]=(Y[o*64+s]-m)/sd;}return Y;}
const relu=Y=>{for(let i=0;i<Y.length;i++)if(Y[i]<0)Y[i]=0;return Y;};
let NET=null;
function runNet(){
  const X=buildInput(S.pos);
  const Xs=X.slice();for(let s=0;s<64;s++){Xs[113*64+s]/=50;Xs[118*64+s]/=50;}
  const c1=relu(norm(conv(Xs,119,W1,256),256));
  const maps=[c1.slice(0,TC*64)];let x=maps[0];
  for(let b=0;b<19;b++){const y=relu(norm(conv(x,TC,TW[b][0],TC),TC));const z=norm(conv(y,TC,TW[b][1],TC),TC);
    const n=new Float32Array(TC*64);for(let i=0;i<n.length;i++)n[i]=Math.max(0,x[i]+z[i]);maps.push(n);x=n;}
  const vp=new Float32Array(64);for(let s=0;s<64;s++){let a=0;for(let o=0;o<TC;o++)a+=VW[o]*x[o*64+s];vp[s]=Math.max(0,a);}
  NET={X,c1,maps,vp};
}

/* ================= Filter ================= */
function filterAt(r,c){
  const B=G.pos[S.pos].b,A=S.variant==="A";
  const pawn=A?"P":"p",rr=A?r-1:r+1,terms=[];let sum=-1;
  for(const dc of[-1,1]){const cc=c+dc;if(rr<0||rr>7||cc<0||cc>7){terms.push({sq:null});continue;}
    const hit=B[rr][cc]===pawn;if(hit)sum+=0.5;terms.push({sq:sqName(rr,cc),hit});}
  const ch=B[r][c];const tgt=!!ch&&(A?ch===ch.toLowerCase():ch===ch.toUpperCase());if(tgt)sum+=1;
  return {terms,tgt,ch,sum,out:Math.max(0,sum)};
}
const SWEEP=[];for(let r=7;r>=0;r--)for(let c=0;c<8;c++)SWEEP.push({r,c});

/* ================= Policy encoding ================= */
const DIRS=[[1,0,"north"],[1,1,"north-east"],[0,1,"east"],[-1,1,"south-east"],[-1,0,"south"],[-1,-1,"south-west"],[0,-1,"west"],[1,-1,"north-west"]];
const KN=[[2,1],[1,2],[-1,2],[-2,1],[-2,-1],[-1,-2],[1,-2],[2,-1]];
function polInfo(k,sideW=true){
  if(k<56){const d=Math.floor(k/7),n=k%7+1;return{name:"Slide "+DIRS[d][2]+", "+n+" square"+(n>1?"s":""),dr:DIRS[d][0]*n,dc:DIRS[d][1]*n};}
  if(k<64){const [dr,dc]=KN[k-56];return{name:"Knight jump ("+(dr>0?"+":"")+dr+" rank, "+(dc>0?"+":"")+dc+" file)",dr,dc};}
  const j=k-64,d=Math.floor(j/3),p=j%3;return{name:"Promote to "+["knight","bishop","rook"][p]+", "+["toward the a-file","straight ahead","toward the h-file"][d],dr:sideW?1:-1,dc:d-1};
}
function planeOf(m){
  const a=sqRC(m.from),b=sqRC(m.to),dr=b.r-a.r,dc=b.c-a.c;
  if(m.promotion&&m.promotion!=="q")return 64+(dc+1)*3+["n","b","r"].indexOf(m.promotion);
  const adr=Math.abs(dr),adc=Math.abs(dc);
  if((adr===1&&adc===2)||(adr===2&&adc===1))return 56+KN.findIndex(k=>k[0]===dr&&k[1]===dc);
  const d=DIRS.findIndex(x=>x[0]===Math.sign(dr)&&x[1]===Math.sign(dc));return d*7+Math.max(adr,adc)-1;
}

/* ================= Value stand-in ================= */
const MAT={p:1,n:3,b:3,r:5,q:9,k:0};
function valueOf(i){
  const L=G.legal[i],P=G.pos[i];
  if(L.mate)return{v:-1,why:"checkmate"};if(L.draw)return{v:0,why:"draw"};
  let w=0,b=0;for(let r=0;r<8;r++)for(let c=0;c<8;c++){const ch=P.b[r][c];if(!ch)continue;if(ch===ch.toUpperCase())w+=MAT[ch.toLowerCase()];else b+=MAT[ch];}
  const d=P.side==="w"?w-b:b-w;return{v:Math.tanh(d/5),w,b,d,why:"material"};
}

/* ================= Colors ================= */
const C={};
function cssVar(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
function readColors(){["bg","surface","fg","muted","grid","accent","white","black","const","sqLight","sqDark","pieceW","pieceB"].forEach(k=>C[k]=new THREE.Color(cssVar("--"+k)||"#888"));}
const hex=c=>"#"+c.getHexString();

/* ================= Three setup ================= */
const canvas=$("#gl"),stageEl=$("#stage");
let renderer;
try{renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:false});}catch(e){const d=document.createElement("div");d.className="glerr";d.textContent="This view needs WebGL, which isn't available in this browser. The guide still explains every stage.";stageEl.appendChild(d);return;}
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
readColors();
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(42,1,0.1,500);
const controls=new THREE.OrbitControls(camera,canvas);
controls.enableDamping=true;controls.dampingFactor=0.08;

const dummy=new THREE.Object3D(),tmpC=new THREE.Color();
function frameGeo(xs,y0=0){
  const v=[];for(const x of xs){const a=[[y0,-4],[y0+8,-4],[y0+8,4],[y0,4]];for(let i=0;i<4;i++){const p=a[i],q=a[(i+1)%4];v.push(x,p[0],p[1],x,q[0],q[1]);}}
  const g=new THREE.BufferGeometry();g.setAttribute("position",new THREE.Float32BufferAttribute(v,3));return g;}
const GRIDLINES=[];
function lineMat(color,op){return new THREE.LineBasicMaterial({color,transparent:true,opacity:op});}
function gridLines(geo,op){const l=new THREE.LineSegments(geo,lineMat(C.grid,op));scene.add(l);GRIDLINES.push(l);return l;}
function cellMesh(n,th,h=0.84){const m=new THREE.InstancedMesh(new THREE.BoxGeometry(th,h,h),new THREE.MeshBasicMaterial(),n);
  for(let i=0;i<n;i++)m.setColorAt(i,C.fg);m.count=0;scene.add(m);return m;}
function setCell(m,i,x,y,z,s,col){dummy.position.set(x,y,z);dummy.rotation.set(0,0,0);dummy.scale.set(1,s,s);dummy.updateMatrix();m.setMatrixAt(i,dummy.matrix);m.setColorAt(i,col);}
function flush(m,n){m.count=n;m.instanceMatrix.needsUpdate=true;if(m.instanceColor)m.instanceColor.needsUpdate=true;}
const cy=r=>r+0.5,cz=c=>c-3.5;
const clamp01=x=>Math.max(0,Math.min(1,x));
const easeIO=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
const easeBack=t=>{const c1=1.9,c3=c1+1;return 1+c3*Math.pow(t-1,3)+c1*Math.pow(t-1,2);};
const easeElastic=t=>t<=0?0:t>=1?1:Math.pow(2,-9*t)*Math.sin((t*10-0.75)*(2*Math.PI)/3)+1;

/* Labels */
const LABELS=[];
function label(text,pos,size=0.7,muted=false){
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({transparent:true,depthTest:false}));
  sp.position.copy(pos);sp.renderOrder=20;scene.add(sp);const L={sp,text,size,muted};LABELS.push(L);drawLabel(L);return L;}
function drawLabel(L){
  const cv=document.createElement("canvas"),x=cv.getContext("2d"),fs=64;
  const font=(L.muted?"500 ":"700 ")+fs+"px "+(L.muted?'"JetBrains Mono", monospace':'"Familjen Grotesk", Arial, sans-serif');
  x.font=font;const w=Math.ceil(x.measureText(L.text).width)+24;cv.width=w;cv.height=fs*1.4;
  x.font=font;x.fillStyle=hex(L.muted?C.muted:C.fg);x.textBaseline="middle";x.fillText(L.text,12,cv.height/2);
  if(L.sp.material.map)L.sp.material.map.dispose();
  const t=new THREE.CanvasTexture(cv);t.minFilter=THREE.LinearFilter;L.sp.material.map=t;L.sp.material.needsUpdate=true;
  L.sp.scale.set(L.size*w/cv.height,L.size,1);
}

/* ================= Board ================= */
const boardCv=document.createElement("canvas");boardCv.width=boardCv.height=512;
const boardTex=new THREE.CanvasTexture(boardCv);
const boardMesh=new THREE.Mesh(new THREE.PlaneGeometry(8,8),new THREE.MeshBasicMaterial({map:boardTex,side:THREE.DoubleSide}));
boardMesh.geometry.rotateY(-Math.PI/2);boardMesh.position.set(0,4,0);scene.add(boardMesh);
const boardBack=new THREE.Mesh(new THREE.BoxGeometry(0.12,8.3,8.3),new THREE.MeshBasicMaterial());boardBack.position.set(0.08,4,0);scene.add(boardBack);
const GLYPH={k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
const Q=64;
function drawPiece(x,ch,px,py,scale=1,alpha=1){
  if(alpha<=0)return;const w=ch===ch.toUpperCase();
  x.save();x.globalAlpha=alpha;x.translate(px,py);x.scale(scale,scale);
  x.textAlign="center";x.textBaseline="middle";x.font="46px 'DejaVu Sans','Segoe UI Symbol','Noto Sans Symbols 2',serif";
  const g=GLYPH[ch.toLowerCase()]+"︎";
  x.lineWidth=3;x.strokeStyle=hex(w?C.pieceB:C.pieceW);x.strokeText(g,0,3);x.fillStyle=hex(w?C.pieceW:C.pieceB);x.fillText(g,0,3);x.restore();
}
function movingPieces(mv){
  const a=sqRC(mv.from),b=sqRC(mv.to),ch=mv.color==="w"?mv.piece.toUpperCase():mv.piece;
  const list=[{ch,a,b,promo:mv.promotion?(mv.color==="w"?mv.promotion.toUpperCase():mv.promotion):null}];
  if(mv.flags.includes("k"))list.push({ch:mv.color==="w"?"R":"r",a:{r:a.r,c:7},b:{r:a.r,c:5}});
  if(mv.flags.includes("q"))list.push({ch:mv.color==="w"?"R":"r",a:{r:a.r,c:0},b:{r:a.r,c:3}});
  let cap=null;if(mv.flags.includes("e"))cap={r:a.r,c:b.c};else if(mv.captured)cap=b;
  return {list,cap};
}
function drawBoard(k=1){
  const x=boardCv.getContext("2d");
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){x.fillStyle=hex((r+c)%2?C.sqLight:C.sqDark);x.fillRect(c*Q,(7-r)*Q,Q,Q);}
  const ov=overlay();
  for(const o of ov.cells){x.fillStyle=o.color;x.globalAlpha=o.a;x.fillRect(o.c*Q,(7-o.r)*Q,Q,Q);x.globalAlpha=1;
    if(o.ring){x.strokeStyle=o.color;x.lineWidth=5;x.strokeRect(o.c*Q+2.5,(7-o.r)*Q+2.5,Q-5,Q-5);}}
  x.font="10px 'JetBrains Mono', monospace";x.fillStyle=hex(C.muted);x.textBaseline="alphabetic";x.textAlign="start";
  for(let c=0;c<8;c++)x.fillText(FILES[c],c*Q+Q-10,Q*8-4);for(let r=0;r<8;r++)x.fillText(r+1,3,(7-r)*Q+12);
  const animMv=ANIM&&k<1?G.moves[S.pos]:null;
  if(animMv){
    const P0=G.pos[S.pos-1].b,{list,cap}=movingPieces(animMv),skip=new Set(list.map(m=>m.a.r*8+m.a.c));if(cap)skip.add(cap.r*8+cap.c);
    for(let r=0;r<8;r++)for(let c=0;c<8;c++){const ch=P0[r][c];if(ch&&!skip.has(r*8+c))drawPiece(x,ch,c*Q+Q/2,(7-r)*Q+Q/2);}
    if(cap){const ch=P0[cap.r][cap.c];drawPiece(x,ch,cap.c*Q+Q/2,(7-cap.r)*Q+Q/2,1+0.4*k,1-easeIO(clamp01(k*1.6-0.4)));}
    const e=easeIO(k);
    for(const m of list){const px=(m.a.c+(m.b.c-m.a.c)*e)*Q+Q/2,py=(7-(m.a.r+(m.b.r-m.a.r)*e))*Q+Q/2-Math.sin(Math.PI*e)*18;
      drawPiece(x,k>0.85&&m.promo?m.promo:m.ch,px,py,1+0.3*Math.sin(Math.PI*e));}
  }else{const P=G.pos[S.pos].b;for(let r=0;r<8;r++)for(let c=0;c<8;c++){const ch=P[r][c];if(ch)drawPiece(x,ch,c*Q+Q/2,(7-r)*Q+Q/2);}}
  for(const d of ov.dots){x.fillStyle=hex(C.accent);x.globalAlpha=0.85;x.beginPath();x.arc(d.c*Q+Q/2,(7-d.r)*Q+Q/2,d.cap?26:10,0,Math.PI*2);
    if(d.cap){x.lineWidth=5;x.strokeStyle=hex(C.accent);x.stroke();}else x.fill();x.globalAlpha=1;}
  if(ov.arrow){const a=ov.arrow,fx=a.c0*Q+Q/2,fy=(7-a.r0)*Q+Q/2,tx=a.c1*Q+Q/2,ty=(7-a.r1)*Q+Q/2,ang=Math.atan2(ty-fy,tx-fx);
    x.strokeStyle=x.fillStyle=hex(a.color||C.accent);x.globalAlpha=0.9;x.lineWidth=9;x.lineCap="round";x.beginPath();x.moveTo(fx,fy);x.lineTo(tx-Math.cos(ang)*18,ty-Math.sin(ang)*18);x.stroke();
    x.beginPath();x.moveTo(tx,ty);x.lineTo(tx-Math.cos(ang-0.5)*28,ty-Math.sin(ang-0.5)*28);x.lineTo(tx-Math.cos(ang+0.5)*28,ty-Math.sin(ang+0.5)*28);x.closePath();x.fill();x.globalAlpha=1;}
  boardTex.needsUpdate=true;
  const pb=$("#pboard").getContext("2d");pb.drawImage(boardCv,0,0);
}
function overlay(){
  const A=hex(C.accent),W=hex(C.white),cells=[],dots=[];let arrow=null;
  const last=G.moves[S.pos];
  if(last&&(S.stage==="play"||S.stage==="value")){[last.from,last.to].forEach(s=>{const p=sqRC(s);cells.push({r:p.r,c:p.c,color:W,a:0.32});});}
  if(S.stage==="play"&&S.sel){cells.push({r:S.sel.r,c:S.sel.c,color:A,a:0.4,ring:true});
    for(const m of G.legal[S.pos].moves)if(m.from===sqName(S.sel.r,S.sel.c)){const p=sqRC(m.to);if(!dots.some(d=>d.r===p.r&&d.c===p.c))dots.push({r:p.r,c:p.c,cap:!!m.captured});}}
  if(S.stage==="input"&&S.plane<112&&!ANIM){for(let s=0;s<64;s++)if(NET.X[S.plane*64+s])cells.push({r:s>>3,c:s&7,color:A,a:0.35,ring:true});}
  if(S.stage==="filter"){const {r,c}=S.cur;for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++){const rr=r+dr,cc=c+dc;if(rr<0||rr>7||cc<0||cc>7)continue;cells.push({r:rr,c:cc,color:A,a:dr||dc?0.22:0.5,ring:!dr&&!dc});}}
  if(S.stage==="tower"){const h=Math.floor(rfSize(S.layer)/2),{r,c}=S.rf;
    for(let rr=Math.max(0,r-h);rr<=Math.min(7,r+h);rr++)for(let cc=Math.max(0,c-h);cc<=Math.min(7,c+h);cc++)cells.push({r:rr,c:cc,color:A,a:rr===r&&cc===c?0.55:0.22,ring:rr===r&&cc===c});}
  if(S.stage==="policy"){const {r,c}=S.from,I=polInfo(S.pol,G.pos[S.pos].side==="w");cells.push({r,c,color:A,a:0.35,ring:true});
    const r1=r+I.dr,c1=c+I.dc;if(r1>=0&&r1<8&&c1>=0&&c1<8)arrow={r0:r,c0:c,r1,c1};}
  return {cells,dots,arrow};
}

/* ================= Scene layout ================= */
const IX=i=>i<112?3+i*0.1+Math.floor(i/14)*0.3:3+111*0.1+7*0.3+0.5+(i-112)*0.25;
const GROUPW=14*0.1+0.3;
const FX=20.2,CX=o=>22+o*0.05,TX=b=>37.6+b*1.2,PX=k=>64+k*0.09,PY=6,VY=-6;

/* Input stack */
const inputMesh=cellMesh(119*64,0.07);
const inputMap=new Int16Array(119*64);
gridLines(frameGeo([...Array(119)].map((_,i)=>IX(i))),0.45);
const selFrame=new THREE.LineSegments(frameGeo([0]),lineMat(C.accent,1));scene.add(selFrame);
let filtFrames=null;
function inputColor(i){if(i<112){const t=Math.floor(i/14),j=i%14;return (j<6?C.white:j<12?C.black:C.muted).clone().lerp(C.bg,t*0.085);}return C.const;}
function updateInput(shift=1){
  const X=NET.X,e=easeBack(clamp01(shift)),slide=easeIO(clamp01(shift*1.4));let n=0;
  for(let i=0;i<119;i++){
    const col=inputColor(i),t=i<112?Math.floor(i/14):-1;
    const x=t>=1?IX(i)-(1-slide)*GROUPW:IX(i);
    const s=t===0||t===-1?Math.max(0,e):1;if(s<=0.01)continue;
    for(let q=0;q<64;q++){if(!X[i*64+q])continue;setCell(inputMesh,n,x,cy(q>>3),cz(q&7),s,col);inputMap[n]=i;n++;}}
  flush(inputMesh,n);
}

/* Filter output plane + window */
const filtMesh=cellMesh(64,0.06);
gridLines(frameGeo([FX]),0.9);
const winSpan=IX(118)-IX(0)+0.4;
const winBox=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(winSpan,3,3)),lineMat(C.accent,0.9));scene.add(winBox);
const outCursor=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.25,1,1)),lineMat(C.accent,1));scene.add(outCursor);
function updateFilter(pop=1){
  let n=0;const s=easeBack(clamp01(pop));
  for(let k=0;k<S.revealed;k++){const {r,c}=SWEEP[k],f=filterAt(r,c);if(f.out<=0)continue;
    setCell(filtMesh,n++,FX,cy(r),cz(c),Math.max(0.01,(0.6+0.4*Math.min(1,f.out))*s),C.accent);}
  flush(filtMesh,n);
  winBox.position.set((IX(0)+IX(118))/2,cy(S.cur.r),cz(S.cur.c));outCursor.position.set(FX,cy(S.cur.r),cz(S.cur.c));
  const reads=S.variant==="A"?[0,6,7,8,9,10,11]:[6,0,1,2,3,4,5];
  if(filtFrames){scene.remove(filtFrames);filtFrames.geometry.dispose();}
  filtFrames=new THREE.LineSegments(frameGeo(reads.map(IX)),lineMat(C.accent,0.9));scene.add(filtFrames);
  filtFrames.visible=S.stage==="filter";
}

/* Conv1 stack */
const c1Mesh=cellMesh(256*64,0.03);
gridLines(frameGeo([...Array(256)].map((_,o)=>CX(o)).filter((_,o)=>o%8===0||o===255)),0.4);
const c1Box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(CX(255)-CX(0)+0.2,8,8)),lineMat(C.grid,0.7));
c1Box.position.set((CX(0)+CX(255))/2,4,0);scene.add(c1Box);
function updateC1(wave=2){
  let n=0;const Y=NET.c1;
  for(let o=0;o<256;o++){const local=clamp01((wave-o/256*0.6)/0.4);if(local<=0)continue;const pop=easeBack(local);
    for(let s=0;s<64;s++){const v=Y[o*64+s];if(v<1.1)continue;const t=Math.min(1,(v-1.1)/1.6);
      tmpC.copy(C.bg).lerp(C.accent,0.35+0.65*t);setCell(c1Mesh,n++,CX(o),cy(s>>3),cz(s&7),Math.max(0.01,(0.45+0.5*t)*pop),tmpC);}}
  flush(c1Mesh,n);
}

/* Tower */
const BLOCKS=[];
for(let b=0;b<19;b++){
  const g=new THREE.Group();const x=TX(b)+0.35;
  const box=new THREE.Mesh(new THREE.BoxGeometry(0.7,8,8),new THREE.MeshBasicMaterial({transparent:true,opacity:0.07,depthWrite:false}));
  const edges=new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry),lineMat(C.grid,0.8));
  const data=new Uint8Array(64*4);const tex=new THREE.DataTexture(data,8,8,THREE.RGBAFormat);tex.magFilter=tex.minFilter=THREE.NearestFilter;
  const face=new THREE.Mesh(new THREE.PlaneGeometry(7.9,7.9),new THREE.MeshBasicMaterial({map:tex,side:THREE.DoubleSide}));
  face.geometry.rotateY(-Math.PI/2);face.position.set(-0.36,0,0);
  g.add(box,edges,face);g.position.set(x,4,0);scene.add(g);
  const pts=[];for(let i=0;i<=16;i++){const t=i/16;pts.push(new THREE.Vector3(TX(b)-0.15+t*1.0,8.2+Math.sin(t*Math.PI)*0.9,0));}
  const arc=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),lineMat(C.muted,0.7));scene.add(arc);
  BLOCKS.push({g,box,edges,tex,data,arc,pulse:0,shown:-1});
}
function heatFill(data,map,ch){
  let mx=1e-6;for(let s=0;s<64;s++)mx=Math.max(mx,map[ch*64+s]);
  for(let s=0;s<64;s++){const t=Math.pow(map[ch*64+s]/mx,0.8);tmpC.copy(C.bg).lerp(C.accent,t);
    data[s*4]=tmpC.r*255;data[s*4+1]=tmpC.g*255;data[s*4+2]=tmpC.b*255;data[s*4+3]=255;}
}
function blankFill(data){for(let s=0;s<64;s++){data[s*4]=C.bg.r*255;data[s*4+1]=C.bg.g*255;data[s*4+2]=C.bg.b*255;data[s*4+3]=255;}}
function styleBlocks(){
  BLOCKS.forEach((B,b)=>{
    const sel=S.stage==="tower"&&S.layer===b+1,p=B.pulse;
    B.box.material.color.copy(sel||p>0?C.accent:C.fg);B.box.material.opacity=sel?0.22:0.06+0.4*p;
    B.edges.material.color.copy(sel||p>0.05?C.accent:C.grid);B.edges.material.opacity=sel?1:0.8;
    B.arc.material.color.copy(p>0.05?C.accent:C.muted);B.arc.material.opacity=0.7+0.3*p;
    B.g.scale.set(1+1.1*p,1+0.07*p,1+0.07*p);B.arc.position.y=0.6*p;});
  const c1sel=S.stage==="tower"&&S.layer===0;c1Box.material.color.copy(c1sel?C.accent:C.grid);c1Box.material.opacity=c1sel?1:0.7;
}
function updateTower(){
  BLOCKS.forEach((B,b)=>{heatFill(B.data,NET.maps[b+1],S.chan);B.tex.needsUpdate=true;B.shown=S.pos;});
  styleBlocks();
  const mini=$("#mini"),x=mini.getContext("2d"),M=NET.maps[S.layer];let mx=1e-6;for(let s=0;s<64;s++)mx=Math.max(mx,M[S.chan*64+s]);
  for(let s=0;s<64;s++){const r=s>>3,c=s&7,t=Math.pow(M[S.chan*64+s]/mx,0.8);x.fillStyle=hex(tmpC.copy(C.bg).lerp(C.accent,t));x.fillRect(c*20,(7-r)*20,20,20);}
  x.strokeStyle=hex(C.fg);x.lineWidth=2;x.strokeRect(S.rf.c*20+1,(7-S.rf.r)*20+1,18,18);
}
const rfSize=L=>3+4*L;

/* Policy head */
gridLines(frameGeo([...Array(73)].map((_,k)=>PX(k)).filter((_,k)=>k%8===0||k===72),PY),0.45);
const polBox=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(PX(72)-PX(0)+0.2,8,8)),lineMat(C.grid,0.7));polBox.position.set((PX(0)+PX(72))/2,PY+4,0);scene.add(polBox);GRIDLINES.push(polBox);
const polSel=new THREE.LineSegments(frameGeo([0],PY),lineMat(C.accent,1));scene.add(polSel);
const polMesh=cellMesh(73*64,0.07);
const polMap=[];
function updatePolicy(pop=1){
  const L=G.legal[S.pos].moves;let n=0;polMap.length=0;
  L.forEach((m,i)=>{const p=sqRC(m.from),k=planeOf(m),local=clamp01((pop-i/Math.max(1,L.length)*0.5)/0.5);if(local<=0)return;
    const selm=k===S.pol&&p.r===S.from.r&&p.c===S.from.c;
    setCell(polMesh,n,PX(k),PY+cy(p.r),cz(p.c),Math.max(0.01,easeBack(local)*(selm?1.1:0.8)),selm?C.fg:C.accent);polMap[n]=i;n++;});
  flush(polMesh,n);polSel.position.x=PX(S.pol);
}

/* Value head */
const vMesh=cellMesh(64,0.06);
gridLines(frameGeo([64],VY),0.9);
const fcMesh=cellMesh(256,0.3,0.3);
const gaugeTrack=new THREE.Mesh(new THREE.BoxGeometry(0.25,8,0.25),new THREE.MeshBasicMaterial());gaugeTrack.position.set(71,VY+4,0);scene.add(gaugeTrack);
const needle=new THREE.Mesh(new THREE.BoxGeometry(0.9,0.18,0.9),new THREE.MeshBasicMaterial());scene.add(needle);
const zeroTick=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.05,0.5),new THREE.MeshBasicMaterial());zeroTick.position.set(71,VY+4,0);scene.add(zeroTick);
let needleV=0;
function updateValue(pop=1){
  let mx=1e-6,n=0;for(let s=0;s<64;s++)mx=Math.max(mx,NET.vp[s]);const e=easeBack(clamp01(pop*2));
  for(let s=0;s<64;s++){const t=NET.vp[s]/mx;if(t<0.15)continue;tmpC.copy(C.bg).lerp(C.accent,0.3+0.7*t);setCell(vMesh,n++,64,VY+cy(s>>3),cz(s&7),Math.max(0.01,(0.5+0.4*t)*e),tmpC);}
  flush(vMesh,n);
  const fe=clamp01(pop*2-0.5);
  for(let i=0;i<256;i++){const r=i>>4,c=i&15,lit=0.25+0.5*Math.abs(Math.sin(i*12.9898+S.pos*3.1));tmpC.copy(C.bg).lerp(C.accent,lit*(0.3+0.7*fe));
    dummy.position.set(67.5,VY+0.4+r*0.48,-3.6+c*0.48);dummy.scale.set(1,1,1);dummy.updateMatrix();fcMesh.setMatrixAt(i,dummy.matrix);fcMesh.setColorAt(i,tmpC);}
  flush(fcMesh,256);
  gaugeTrack.material.color.copy(C.grid);zeroTick.material.color.copy(C.muted);needle.material.color.copy(C.accent);
  needle.position.set(71,VY+4+needleV*4,0);
}

/* Connectors + signal orbs */
const CONN=[];
function poly(pts){const l=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map(p=>new THREE.Vector3(...p))),lineMat(C.muted,0.6));scene.add(l);CONN.push(l);return l;}
poly([[0.3,4,0],[2.7,4,0]]);poly([[IX(118)+0.3,4,0],[FX-0.3,4,0]]);poly([[FX+0.3,4,0],[CX(0)-0.3,4,0]]);
poly([[CX(255)+0.3,4,0],[TX(0)-0.3,4,0]]);poly([[TX(18)+0.9,4,0],[62,4,0],[63.7,PY+4,0]]);poly([[62,4,0],[63.7,VY+4,0]]);
poly([[64.2,VY+4,0],[67.2,VY+4,0]]);poly([[67.8,VY+4,0],[70.7,VY+4,0]]);
function glowTex(){const cv=document.createElement("canvas");cv.width=cv.height=128;const x=cv.getContext("2d");const g=x.createRadialGradient(64,64,0,64,64,64);
  g.addColorStop(0,"rgba(255,255,255,1)");g.addColorStop(0.25,"rgba(255,255,255,.85)");g.addColorStop(1,"rgba(255,255,255,0)");x.fillStyle=g;x.fillRect(0,0,128,128);return new THREE.CanvasTexture(cv);}
const GT=glowTex();
function orb(){const s=new THREE.Sprite(new THREE.SpriteMaterial({map:GT,transparent:true,depthTest:false}));s.scale.set(2.2,2.2,1);s.renderOrder=30;s.visible=false;scene.add(s);return s;}
const ORBS=[orb(),orb()];

/* 3D labels */
label("Board",new THREE.Vector3(0,9.3,0));
label("Input · 8×8×119",new THREE.Vector3((IX(0)+IX(118))/2,9.3,0));
label("Filter output",new THREE.Vector3(FX,9.3,0),0.55);
label("Conv · 8×8×256",new THREE.Vector3((CX(0)+CX(255))/2,9.3,0));
label("Residual tower · 19 blocks",new THREE.Vector3((TX(0)+TX(18))/2,10.6,0));
label("Policy · 8×8×73",new THREE.Vector3((PX(0)+PX(72))/2,PY+9.3,0));
label("Value",new THREE.Vector3(67.5,VY+9.3,0));
label("+1 win",new THREE.Vector3(72.6,VY+8,0),0.45,true);
label("0 draw",new THREE.Vector3(72.6,VY+4,0),0.45,true);
label("−1 loss",new THREE.Vector3(72.6,VY,0),0.45,true);
label("now",new THREE.Vector3(IX(6),-0.6,0),0.4,true);
label("7 moves ago",new THREE.Vector3(IX(104),-0.6,0),0.4,true);
label("game state",new THREE.Vector3(IX(115),-0.6,0),0.4,true);
const selLabel=label(" ",new THREE.Vector3(0,10.4,0),0.5,true);

/* ================= Camera ================= */
const VIEWS={
  play:{p:[28,27,90],t:[36,2,0]},
  input:{p:[-9,15,28],t:[9,4,0]},
  filter:{p:[-6,15,32],t:[11,4,0]},
  tower:{p:[27,20,44],t:[47,4,0]},
  policy:{p:[50,24,35],t:[67,PY+3,0]},
  value:{p:[52,6,32],t:[68,VY+4,0]}
};
let fly=null;
function flyTo(v){const P=new THREE.Vector3(...v.p),T=new THREE.Vector3(...v.t);
  if(reduced){camera.position.copy(P);controls.target.copy(T);return;}
  fly={p0:camera.position.clone(),t0:controls.target.clone(),P,T,s:performance.now()};}

/* ================= The move animation ================= */
/* Timeline in seconds. Each stage answers in turn, like a relay. */
const TL={board:[0,0.6],input:[0.5,1.1],filter:[1.05,1.35],conv:[1.25,1.85],tower:[1.75,2.85],policy:[2.8,3.4],value:[2.8,3.7]};
const TOTAL=3.8;
const ORBPATH=[[0.55,[0.5,4,0]],[1.1,[IX(118),4,0]],[1.3,[FX,4,0]],[1.85,[CX(255),4,0]],[2.85,[TX(18)+0.9,4,0]],[2.95,[62,4,0]]];
let ANIM=null;
const prog=(t,w)=>clamp01((t-w[0])/(w[1]-w[0]));
function orbAt(t){
  if(t<ORBPATH[0][0]||t>3.25)return null;
  for(let i=0;i<ORBPATH.length-1;i++){const [t0,a]=ORBPATH[i],[t1,b]=ORBPATH[i+1];if(t<=t1){const e=(t-t0)/(t1-t0);return a.map((v,k)=>v+(b[k]-v)*e);}}
  return null;
}
function startMoveAnim(){
  const vFrom=needleV;S.sel=null;runNet();
  ANIM={t0:performance.now(),vFrom,vTo:valueOf(S.pos).v};
  BLOCKS.forEach(B=>{B.pulse=0;blankFill(B.data);B.tex.needsUpdate=true;B.shown=-1;});
  S.revealed=64;
  refreshPanels();
}
function stepAnim(now){
  const t=(now-ANIM.t0)/1000;
  drawBoard(prog(t,TL.board));
  updateInput(prog(t,TL.input));
  updateFilter(prog(t,TL.filter));
  updateC1(prog(t,TL.conv)*1.0);
  BLOCKS.forEach((B,b)=>{const st=TL.tower[0]+b*0.045,p=clamp01((t-st)/0.32);B.pulse=p>0&&p<1?Math.sin(Math.PI*p):0;
    if(p>0.3&&B.shown!==S.pos){heatFill(B.data,NET.maps[b+1],S.chan);B.tex.needsUpdate=true;B.shown=S.pos;}});
  styleBlocks();
  updatePolicy(prog(t,TL.policy));
  const vp=prog(t,TL.value);needleV=ANIM.vFrom+(ANIM.vTo-ANIM.vFrom)*easeElastic(vp);updateValue(vp);
  const o=orbAt(t);
  if(o){ORBS[0].visible=true;ORBS[0].position.set(...o);ORBS[1].visible=false;}
  else if(t>2.95&&t<3.35){const e=easeIO((t-2.95)/0.4);ORBS[0].visible=ORBS[1].visible=true;
    ORBS[0].position.set(62+(64-62)*e,4+(PY+4-4)*e,0);ORBS[1].position.set(62+(71-62)*e,4+(VY+4-4)*e,0);}
  else ORBS[0].visible=ORBS[1].visible=false;
  const s=ORBS[0].visible?1.6+0.6*Math.sin(t*18):0;ORBS.forEach(q=>{q.scale.set(2.2+s*0.3,2.2+s*0.3,1);q.material.color.copy(C.accent);});
  document.querySelectorAll("#signal span").forEach(el=>{const w=TL[el.dataset.k==="conv"?"conv":el.dataset.k];const p=prog(t,w);el.classList.toggle("on",p>0&&p<1);el.classList.toggle("done",p>=1);});
  if(S.follow){const f=o||(ORBS[0].visible?ORBS[0].position.toArray():null);if(f){const T=new THREE.Vector3(f[0],4,0);controls.target.lerp(T,0.08);camera.position.lerp(T.clone().add(new THREE.Vector3(-9,9,19)),0.06);}}
  if(t>=TOTAL){ANIM=null;BLOCKS.forEach(B=>B.pulse=0);ORBS.forEach(q=>q.visible=false);
    document.querySelectorAll("#signal span").forEach(el=>el.classList.remove("on","done"));
    needleV=valueOf(S.pos).v;refresh();
    if(S.playing){if(S.pos<G.fens.length-1)autoTimer=setTimeout(()=>{if(S.playing)goTo(S.pos+1,true);},650);else{S.playing=false;refreshPanels();}}}
}
let autoTimer=null;

/* ================= Navigation ================= */
function defaultPolicySel(){
  const L=G.legal[S.pos].moves;if(!L.length)return;
  const nx=G.moves[S.pos+1];const m=nx?L.find(x=>x.from===nx.from&&x.to===nx.to)||L[0]:L[0];
  S.from=sqRC(m.from);S.pol=planeOf(m);$("#polSlider").value=S.pol;
}
function goTo(i,animate){
  i=Math.max(0,Math.min(G.fens.length-1,i));
  const forward=i===S.pos+1&&G.moves[i];
  S.pos=i;S.sel=null;defaultPolicySel();$("#posSlider").max=G.fens.length-1;$("#posSlider").value=i;
  if(animate&&forward&&!reduced)startMoveAnim();
  else{ANIM=null;runNet();needleV=valueOf(S.pos).v;S.revealed=64;newStatic();}
}
function newStatic(){updateInput(1);updateC1(2);refresh();}

/* ================= Panels ================= */
function setStage(name){
  S.stage=name;
  document.querySelectorAll(".tabs button").forEach(b=>b.setAttribute("aria-selected",b.dataset.stage===name));
  document.querySelectorAll("[data-sec]").forEach(s=>s.hidden=s.dataset.sec!==name);
  if(name!=="filter")stopSweep();
  $("#hint").textContent=name==="play"?"Drag to orbit · scroll or pinch to zoom · tap a piece to move it":"Drag to orbit · scroll or pinch to zoom · tap the board";
  flyTo(VIEWS[name]);refresh();
}
function refreshVisibility(){
  selFrame.visible=S.stage==="input";selLabel.sp.visible=S.stage==="input";
  winBox.visible=outCursor.visible=S.stage==="filter";if(filtFrames)filtFrames.visible=S.stage==="filter";
  polSel.visible=S.stage==="policy";
}
function refreshPlay(){
  const L=G.legal[S.pos],P=G.pos[S.pos],side=P.side==="w"?"White":"Black";
  let s=moveLabel(S.pos),sub;
  if(L.mate)sub="Checkmate. "+(side==="White"?"Black":"White")+" wins.";else if(L.draw)sub="Draw.";else sub=side+" to move"+(L.check?", in check":"")+" · "+L.moves.length+" legal moves";
  $("#status").innerHTML=s+"<small>"+sub+"</small>";
  $("#playBtn").textContent=S.playing?"❚❚ Pause":"▶ Play";
  $("#backBtn").disabled=S.pos===0;$("#nextBtn").disabled=S.pos>=G.fens.length-1;
  $("#playBtn").disabled=S.pos>=G.fens.length-1&&!S.playing;
}
function refreshInputPanel(){
  const i=S.plane;$("#planeNum").textContent=i+" of 118";
  selFrame.position.x=IX(i);selLabel.sp.position.x=IX(i);selLabel.text=planeName(i);drawLabel(selLabel);
  let ones=0;for(let s=0;s<64;s++)if(NET.X[i*64+s])ones++;
  let h="<span class='k'>Sheet</span> "+planeName(i)+"<br>";
  if(i<112){const t=Math.floor(i/14),q=S.pos-t;
    h+=q<0?"<span class='k'>Moment</span> before the game started, so every square is 0":"<span class='k'>Moment</span> "+(q===0?"the starting position":"after "+moveLabel(q))+"<br><span class='k'>Squares set to 1</span> "+ones+" of 64";}
  else h+="<span class='k'>Value on all 64 squares</span> <span class='hit'>"+NET.X[i*64]+"</span>";
  $("#planeReadout").innerHTML=h;
}
function kernelHTML(){
  const A=S.variant==="A";
  const pawnG=A?[0,0,0,0,0,0,.5,0,.5]:[.5,0,.5,0,0,0,0,0,0];const pg=[0,0,0,0,1,0,0,0,0];
  const grid=w=>'<div class="g">'+w.map(v=>"<span class='"+(v?"on":"")+"'>"+(v||"0")+"</span>").join("")+"</div>";
  return '<div class="kern">'+(A?"White-pawn sheet":"Black-pawn sheet")+grid(pawnG)+'</div><div class="kern">Each '+(A?"Black":"White")+"-piece sheet (×6)"+grid(pg)+'</div><div class="kern">Bias<div class="g" style="grid-template-columns:28px"><span class="on">−1</span></div></div><div class="kern">Other 112 sheets: 0</div>';
}
function refreshFilterPanel(){
  $("#varA").setAttribute("aria-pressed",S.variant==="A");$("#varB").setAttribute("aria-pressed",S.variant==="B");
  $("#kernels").innerHTML=kernelHTML();
  const {r,c}=S.cur,f=filterAt(r,c),A=S.variant==="A";
  let h="<span class='k'>Square</span> "+sqName(r,c)+"<br>";
  f.terms.forEach(t=>{h+="<span class='k'>"+(A?"White":"Black")+" pawn on "+(t.sq||"(off board)")+"</span> "+(t.hit?"<span class='hit'>yes +0.5</span>":"no 0")+"<br>";});
  h+="<span class='k'>"+(A?"Black":"White")+" piece on "+sqName(r,c)+"</span> "+(f.tgt?"<span class='hit'>"+pieceName(f.ch)+" +1</span>":"no 0")+"<br>";
  h+="<span class='k'>Sum + bias</span> "+(f.sum+1).toFixed(1)+" − 1 = "+f.sum.toFixed(1)+"<br><span class='k'>Output</span> "+(f.out>0?"<span class='hit'>"+f.out.toFixed(1)+" fires</span>":"0");
  const hits=[];for(let s=0;s<64;s++){if(filterAt(s>>3,s&7).out>0)hits.push(sqName(s>>3,s&7));}
  h+="<br><span class='k'>Fires on</span> "+(hits.length?"<span class='hit'>"+hits.join(", ")+"</span>":"no squares in this position");
  $("#filterReadout").innerHTML=h;
  $("#sweepBtn").textContent=S.sweeping?"Pause":"Sweep the board";
}
function refreshTowerPanel(){
  const L=S.layer;$("#layerLabel").textContent=L===0?"first convolution":"residual block "+L+" of 19";
  $("#chanLabel").textContent=(S.chan+1)+" of 32 shown";
  const sz=rfSize(L);
  $("#towerReadout").innerHTML="<span class='k'>3×3 convolutions so far</span> "+(1+2*L)+"<br><span class='k'>Patch one square sees</span> "+sz+"×"+sz+"<br>"+(sz>=15?"<span class='hit'>Every square can now see the whole board</span>":"<span class='k'>Whole board from any square at</span> 15×15");
}
function refreshPolicyPanel(){
  const Lg=G.legal[S.pos].moves,nx=G.moves[S.pos+1];
  $("#legalCount").innerHTML="<span class='hit'>"+Lg.length+"</span> legal of 4,672 slots · <span class='k'>masked</span> "+(4672-Lg.length).toLocaleString();
  const ml=$("#moveList");ml.innerHTML="";
  Lg.forEach(m=>{const b=document.createElement("button");b.type="button";b.textContent=m.san;const p=sqRC(m.from),k=planeOf(m);
    b.setAttribute("aria-pressed",k===S.pol&&p.r===S.from.r&&p.c===S.from.c);
    if(nx&&nx.from===m.from&&nx.to===m.to){b.classList.add("next");b.title="Played next in this game";}
    b.addEventListener("click",()=>{S.from=p;S.pol=k;$("#polSlider").value=k;refresh();});ml.appendChild(b);});
  $("#polNum").textContent=S.pol+" of 72";
  const I=polInfo(S.pol,G.pos[S.pos].side==="w"),{r,c}=S.from,r1=r+I.dr,c1=c+I.dc,on=r1>=0&&r1<8&&c1>=0&&c1<8,P=G.pos[S.pos].b[r][c];
  const legal=Lg.find(m=>m.from===sqName(r,c)&&planeOf(m)===S.pol);
  let h="<span class='k'>Sheet</span> "+I.name+"<br><span class='k'>From</span> "+sqName(r,c)+(P?" ("+(P===P.toUpperCase()?"White ":"Black ")+pieceName(P)+")":" (empty)")+"<br>";
  h+="<span class='k'>Lands on</span> "+(on?sqName(r1,c1):"off the board")+"<br><span class='k'>Slot</span> "+((r*8+c)*73+S.pol).toLocaleString()+" of 0–4,671<br>";
  h+="<span class='k'>Status</span> "+(legal?"<span class='hit'>legal: "+legal.san+"</span>":"masked out");
  $("#policyReadout").innerHTML=h;
}
function refreshValuePanel(){
  const V=valueOf(S.pos),side=G.pos[S.pos].side==="w"?"White":"Black";
  $("#meterNeedle").style.left=((V.v+1)/2*100)+"%";
  let h="<span class='k'>Side to move</span> "+side+"<br>";
  if(V.why==="checkmate")h+="<span class='k'>Checkmate</span> <span class='hit'>v = −1</span>";
  else if(V.why==="draw")h+="<span class='k'>Draw</span> <span class='hit'>v = 0</span>";
  else h+="<span class='k'>Material</span> White "+V.w+", Black "+V.b+"<br><span class='k'>v = tanh("+(V.d>0?"+":"")+V.d+" ÷ 5)</span> <span class='hit'>"+(V.v>=0?"+":"")+V.v.toFixed(2)+"</span>";
  h+="<br><span class='k'>Expected score</span> "+((V.v+1)/2*100).toFixed(0)+"%";
  $("#valueReadout").innerHTML=h;
}
function refreshPanels(){
  $("#posLabel").textContent=moveLabel(S.pos);
  refreshPlay();refreshInputPanel();refreshFilterPanel();refreshTowerPanel();refreshPolicyPanel();refreshValuePanel();
}
function refresh(){
  refreshVisibility();refreshPanels();
  if(!ANIM){updateFilter();updateTower();updatePolicy();updateValue();drawBoard();}
}

/* ================= Controls ================= */
document.querySelectorAll(".tabs button").forEach(b=>b.addEventListener("click",()=>setStage(b.dataset.stage)));
$("#posSlider").addEventListener("input",e=>{stopAuto();goTo(+e.target.value,false);});
$("#planeSlider").addEventListener("input",e=>{S.plane=+e.target.value;refresh();});
$("#layerSlider").addEventListener("input",e=>{S.layer=+e.target.value;refresh();});
$("#chanSlider").addEventListener("input",e=>{S.chan=+e.target.value;refresh();});
$("#polSlider").addEventListener("input",e=>{S.pol=+e.target.value;refresh();});
$("#varA").addEventListener("click",()=>{S.variant="A";refresh();});
$("#varB").addEventListener("click",()=>{S.variant="B";refresh();});
$("#follow").addEventListener("change",e=>{S.follow=e.target.checked;if(!S.follow)flyTo(VIEWS[S.stage]);});
function stopAuto(){S.playing=false;clearTimeout(autoTimer);}
$("#playBtn").addEventListener("click",()=>{
  if(S.playing){stopAuto();refreshPanels();return;}
  if(S.pos>=G.fens.length-1)return;S.playing=true;if(!ANIM)goTo(S.pos+1,true);refreshPanels();});
$("#nextBtn").addEventListener("click",()=>{stopAuto();goTo(S.pos+1,true);});
$("#backBtn").addEventListener("click",()=>{stopAuto();goTo(S.pos-1,false);});
$("#operaBtn").addEventListener("click",()=>{stopAuto();loadLine(OPERA);goTo(0,false);});
$("#newBtn").addEventListener("click",()=>{stopAuto();loadLine([]);goTo(0,false);});
let sweepTimer=null;
function stopSweep(){S.sweeping=false;clearInterval(sweepTimer);sweepTimer=null;}
function sweepStep(){if(S.revealed>=64)S.revealed=0;S.cur=SWEEP[S.revealed];S.revealed++;if(S.revealed>=64)stopSweep();refresh();}
$("#stepBtn").addEventListener("click",()=>{stopSweep();sweepStep();});
$("#sweepBtn").addEventListener("click",()=>{if(S.sweeping){stopSweep();refresh();return;}S.sweeping=true;if(S.revealed>=64)S.revealed=0;sweepTimer=setInterval(sweepStep,reduced?400:170);refresh();});

/* Picking */
const ray=new THREE.Raycaster(),ndc=new THREE.Vector2();let down=null;
canvas.addEventListener("pointerdown",e=>{down={x:e.clientX,y:e.clientY};});
canvas.addEventListener("pointerup",e=>{
  if(!down||Math.hypot(e.clientX-down.x,e.clientY-down.y)>6)return;down=null;
  const rc=canvas.getBoundingClientRect();ndc.set((e.clientX-rc.left)/rc.width*2-1,-(e.clientY-rc.top)/rc.height*2+1);
  ray.setFromCamera(ndc,camera);
  const hits=ray.intersectObjects([boardMesh,inputMesh,polMesh]);if(!hits.length)return;const h=hits[0];
  if(h.object===boardMesh){const c=Math.min(7,Math.floor(h.uv.x*8)),r=Math.min(7,Math.floor(h.uv.y*8));
    if(S.stage==="play"){if(ANIM)return;onBoardPlay(r,c);return;}
    if(S.stage==="filter"){stopSweep();S.cur={r,c};S.revealed=64;}
    else if(S.stage==="tower")S.rf={r,c};
    else if(S.stage==="policy")S.from={r,c};
    else if(S.stage==="input"||S.stage==="value"){const ch=G.pos[S.pos].b[r][c];if(ch){S.plane=PIECES.indexOf(ch);$("#planeSlider").value=S.plane;if(S.stage!=="input"){setStage("input");return;}}}
    refresh();}
  else if(h.object===inputMesh&&h.instanceId!=null){S.plane=inputMap[h.instanceId];$("#planeSlider").value=S.plane;if(S.stage!=="input")setStage("input");else refresh();}
  else if(h.object===polMesh&&h.instanceId!=null){const m=G.legal[S.pos].moves[polMap[h.instanceId]];if(m){S.from=sqRC(m.from);S.pol=planeOf(m);$("#polSlider").value=S.pol;if(S.stage!=="policy")setStage("policy");else refresh();}}
});
$("#pboard").addEventListener("click",e=>{if(ANIM)return;const b=e.currentTarget.getBoundingClientRect();const c=Math.min(7,Math.floor((e.clientX-b.left)/b.width*8)),r=7-Math.min(7,Math.floor((e.clientY-b.top)/b.height*8));onBoardPlay(r,c);});
function onBoardPlay(r,c){
  const L=G.legal[S.pos].moves,sq=sqName(r,c);
  if(S.sel){const from=sqName(S.sel.r,S.sel.c),m=L.find(x=>x.from===from&&x.to===sq);
    if(m){stopAuto();if(pushMove({from,to:sq,promotion:"q"}))goTo(S.pos+1,true);return;}}
  S.sel=L.some(x=>x.from===sq)?{r,c}:null;drawBoard();
}

/* ================= Theme ================= */
function applyTheme(){
  readColors();scene.background=C.bg.clone();boardBack.material.color.copy(C.grid);
  GRIDLINES.forEach(l=>l.material.color.copy(C.grid));CONN.forEach(l=>l.material.color.copy(C.muted));
  [selFrame,winBox,outCursor,polSel].forEach(l=>l.material.color.copy(C.accent));
  LABELS.forEach(drawLabel);
  if(NET){updateInput(1);updateC1(2);refresh();}
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
if(document.fonts&&document.fonts.ready)document.fonts.ready.then(()=>{LABELS.forEach(drawLabel);if(!ANIM)drawBoard();});

/* ================= Loop ================= */
function resize(){const w=stageEl.clientWidth,h=stageEl.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
window.__azResize=resize;
new ResizeObserver(resize).observe(stageEl);
function loop(now){
  if(fly){const t=Math.min(1,(now-fly.s)/1100),e=easeIO(t);camera.position.lerpVectors(fly.p0,fly.P,e);controls.target.lerpVectors(fly.t0,fly.T,e);if(t>=1)fly=null;}
  if(ANIM)stepAnim(now);
  controls.update();renderer.render(scene,camera);requestAnimationFrame(loop);
}

loadLine(OPERA);applyTheme();runNet();needleV=valueOf(0).v;defaultPolicySel();newStatic();resize();
$("#posSlider").max=G.fens.length-1;
camera.position.set(...VIEWS.play.p);controls.target.set(...VIEWS.play.t);
requestAnimationFrame(loop);

}
})();
