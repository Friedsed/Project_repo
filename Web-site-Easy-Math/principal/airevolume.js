
function cube(){
    var a=parseFloat(document.getElementById("a").value);
    var perimetredebase=4*a;
    var airedebase= a*a ;
    var   airelaterale= 4*a*a ;
    var airetotale=2*a*a+a*a*4 ;
    var volume=  a*a*a;
    
    
    document.getElementById("perimetredebase").innerHTML= perimetredebase;
    document.getElementById("airedebase").innerHTML=airedebase;
    document.getElementById("airelaterale").innerHTML=airelaterale;
    document.getElementById("airetotale").innerHTML=airetotale;
    document.getElementById("volume").innerHTML=volume;
}


function pave_droit(){
    var lo=parseFloat(document.getElementById("lo").value);
    var la=parseFloat(document.getElementById("la").value);
    var h=parseFloat(document.getElementById("h").value);

   
    var Pbpv = 2*( lo + la ) ;
    var Abpv= lo*la ;
    var Alpv= 2*lo*h+2*la*h ;
    var Atpv= Alpv +2*Abpv ;
    var Vpv=lo*la*h;
    
    document.getElementById("Pbpv").innerHTML= Pbpv;
    document.getElementById("Abpv").innerHTML=Abpv;
    document.getElementById("Alpv").innerHTML=Alpv;
    document.getElementById("Atpv").innerHTML=Atpv;
    document.getElementById("Vpv").innerHTML=Vpv;

}


function cylindre(){
    var r=parseFloat(document.getElementById("r").value);
    var h=parseFloat(document.getElementById("h").value);
   

   
    var Pbc = 2*r*(22/7);
    var Abc= (22/7)*r*r ;
    var Alc= Pbc*h;
    var Atc= Alc+2*Abc ;
    var Vc=Abc*h;
    
    document.getElementById("Pbc").innerHTML= Pbc;
    document.getElementById("Abc").innerHTML=Abc;
    document.getElementById("Alc").innerHTML=Alc;
    document.getElementById("Atc").innerHTML=Atc;
    document.getElementById("Vc").innerHTML=Vc;

}



function cone_revolution(){

    
        var r1=parseFloat(document.getElementById("r1").value);
        var a1=parseFloat(document.getElementById("a1").value);
        var Abcr = r1*r1*(22/7);
        var Pbcr =2*(22/7)*r1;
        var Alcr =r1*a1*(22/7) ;
        var Atcr = Alcr+Abcr;
        var h1=Math.sqrt(a1*a1-r1*r1);
        var Vcr = Abcr*h1/3 ;
    document.getElementById("Pbcr").innerHTML= Pbcr;
    document.getElementById("Abcr").innerHTML=Abcr;
    document.getElementById("Alcr").innerHTML=Alcr;
    document.getElementById("Atcr").innerHTML=Atcr;
    document.getElementById("Vcr").innerHTML=Vcr;
        
}