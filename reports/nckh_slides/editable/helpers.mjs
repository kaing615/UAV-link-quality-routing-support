export const C={bg:'FAFAF8',ink:'0A0A0A',blue:'002FA7',pale:'F0F0EE',line:'D4D4D2',gray:'737373',lightblue:'DEE7FA',white:'FFFFFF'};
export function helpers(slide,deck,record){
 const S=deck.ShapeType;
 const shape=(type,o)=>{record.shapes.push({type,...structuredClone(o)});slide.addShape(type,o);};
 const t=(text,x,y,w,h,size=22,opts={})=>{const o={x,y,w,h,fontFace:'Arial',fontSize:size,color:C.ink,margin:0,breakLine:false,valign:'mid',...opts};record.texts.push({text,...structuredClone(o)});slide.addText(text,o);};
 const rect=(x,y,w,h,fill=C.pale,lineColor=fill,extra={})=>shape(S.rect,{x,y,w,h,fill:{color:fill},line:{color:lineColor,width:1},...extra});
 const line=(x1,y1,x2,y2,color=C.line,width=1.2,dash=false,arrow=false)=>{shape(S.line,{x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1),h:Math.abs(y2-y1),flipH:x2<x1,flipV:y2<y1,line:{color,width,...(dash?{dashType:'dash'}:{}),...(arrow?{endArrowType:'triangle'}:{})}});};
 const arrow=(x1,y1,x2,y2,color=C.blue,width=2.2,dash=false)=>line(x1,y1,x2,y2,color,width,dash,true);
 const dot=(x,y,r=.12,fill=C.blue,outline=fill)=>shape(S.ellipse,{x:x-r,y:y-r,w:2*r,h:2*r,fill:{color:fill},line:{color:outline,width:1.4}});
 const label=(text,x,y,w,h=.48,size=19,fill=C.blue)=>{rect(x,y,w,h,fill);t(text,x+.1,y+.02,w-.2,h-.04,size,{bold:true,color:fill===C.blue?C.white:C.ink,align:'center'});};
 const network=(x,y,w,h,opts={})=>{const pts=opts.pts||[[0,.5],[.28,.08],[.32,.82],[.57,.45],[.8,.08],[1,.5],[.8,.88]];const edges=opts.edges||[[0,1],[0,2],[0,3],[1,3],[1,4],[2,3],[2,6],[3,4],[3,5],[3,6],[4,5],[5,6]];const p=pts.map(([a,b])=>[x+a*w,y+b*h]);for(const [a,b] of edges)line(...p[a],...p[b],opts.lineColor||C.line,1.1,opts.dash||false);for(const [a,b] of opts.path||[])arrow(...p[a],...p[b],C.blue,2.5);p.forEach(([a,b],i)=>dot(a,b,opts.r||.105,(opts.hollow||[]).includes(i)?C.bg:opts.color||C.blue,opts.color||C.blue));return p;};
 const chain=(x,y,w,n=5,opts={})=>{const p=Array.from({length:n},(_,i)=>[x+i*w/(n-1),y]);for(let i=0;i<n-1;i++)line(p[i][0],y,p[i+1][0],y,opts.lineColor||C.line,opts.width||1.5,opts.weak===i);p.forEach(([a,b])=>dot(a,b,opts.r||.14,opts.color||C.blue));return p;};
 const drone=(x,y,scale=1,color=C.blue)=>{line(x-.35*scale,y-.14*scale,x+.35*scale,y+.14*scale,color,2);line(x-.35*scale,y+.14*scale,x+.35*scale,y-.14*scale,color,2);for(const dx of [-.35,.35])for(const dy of [-.14,.14])shape(S.ellipse,{x:x+(dx-.14)*scale,y:y+(dy-.035)*scale,w:.28*scale,h:.07*scale,fill:{color},line:{color,width:.5}});shape(S.ellipse,{x:x-.16*scale,y:y-.1*scale,w:.32*scale,h:.2*scale,fill:{color},line:{color}});line(x-.09*scale,y+.08*scale,x-.15*scale,y+.24*scale,color,1.8);line(x+.09*scale,y+.08*scale,x+.15*scale,y+.24*scale,color,1.8);};
 const circle=(x,y,r,fill=C.pale,outline=fill)=>dot(x,y,r,fill,outline);
 return {C,S,t,rect,line,arrow,dot,label,network,chain,drone,circle,shape,slide,deck};
}
