
function calculate(){
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