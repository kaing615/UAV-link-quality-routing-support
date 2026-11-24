// Slides added 24/11/2026 for the post-hoc checks and v10. Native shapes only.
export function build(key,h,data){
 const {C,t,rect,line,label}=h;
 const b={bold:true};
 const muted={color:C.gray};
 switch(key){
 case 'x1':{
   t('+0,08',.75,2.45,5.5,1.35,84,{bold:true,color:C.blue});
   t('điểm phần trăm',.8,3.82,5.45,.6,32,{bold:true,color:C.blue});
   t('GNN đã làm mượt: 96,07%',.8,4.75,5.6,.48,23,b);
   t('EWMA đã làm mượt: 95,98%',.8,5.32,5.6,.48,23,b);
   line(6.55,2.5,6.55,6.18);
   t('CI 95%: [−0,11; 0,29]',7.1,2.6,5.45,.62,28,b);
   t('Khoảng chứa 0',7.1,3.25,5.4,.55,26,{bold:true,color:C.blue});
   // Axis is linear in percentage points; zero sits inside the interval.
   const zero=8.4,scale=6.5,yy=4.75;
   line(7.15,5.45,12.5,5.45,C.gray,1);
   line(zero,4.05,zero,5.6,C.gray,1.1,true);
   t('0',zero-.2,5.65,.4,.35,18,{align:'center',color:C.gray});
   const lo=zero-.11*scale,hi=zero+.29*scale,est=zero+.08*scale;
   line(lo,yy,hi,yy,C.blue,3);line(lo,yy-.16,lo,yy+.16,C.blue,2);line(hi,yy-.16,hi,yy+.16,C.blue,2);h.dot(est,yy,.115,C.blue);
   t('−0,11',lo-.4,4.98,.8,.32,15,{align:'center',color:C.gray});
   t('0,29',hi-.35,4.98,.7,.32,15,{align:'center',color:C.gray});
   t('Cả hai chọn hệ số làm mượt trên tập tuning riêng: α = β = 0,1',7.1,6.0,5.5,.5,14,muted);
   t('v10 đăng ký trước, 40 kịch bản gradual mới',.75,6.48,5.6,.3,14,muted);
   break;
 }
 case 'x2':{
   // Bars are linear in seconds: 123,43 s maps to 6,2 in.
   const x0=4.1,scale=6.2/123.43;
   const rows=[['Đứng yên',123.43,'123,43 s',C.gray],['Phản ứng',31.45,'31,45 s',C.blue],['Relay lý tưởng',26.09,'26,09 s',C.ink]];
   rows.forEach(([name,v,txt,color],i)=>{const y=2.45+i*1.02;t(name,.75,y,3.2,.55,24,b);rect(x0,y+.07,v*scale,.42,color);t(txt,x0+v*scale+.15,y,1.8,.55,22,{bold:true,color});});
   t('Số giây chia cắt trung bình mỗi lượt',.75,5.55,6.5,.4,18,muted);
   line(.75,5.45,12.6,5.45);
   label('Phản ứng đã đạt khoảng 94% mức giảm có thể có',.75,6.0,8.2,.52,20);
   t('Lý tưởng: relay cách đều,\ndi chuyển tức thời',9.2,5.98,3.4,.58,15,muted);
   t('Thăm dò sau test, 40 kịch bản gradual v9',.75,6.62,6,.3,14,muted);
   break;
 }
 default: throw Error('Unknown extra slide '+key);
 }
}
