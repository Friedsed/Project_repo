function pgcdRec(a,b){
    if(b===0){
        return a;
    }
   return pgcdRec(b,a%b);
}
function ppcmRec(a,b){
    return a*b/pgcdRec(a,b);
}

 var cube=function(a){
    alert("le perimètre de base du cube est " + 4*a);
    alert("l'aire de base du cube est :" + a*a);
    alert("l'aire latérale du cube est :"+ 4*a*a);
    alert("l'aire total du cube est :"+ 2*a*a + a*a*4)
    alert("le volume du cube est :" + a*a*a);

}

var pave_droit=function(la,lo,h){
    alert("le perimètre de base du pave droit est " + 2*(lo+la));
    alert("l'aire de base du pave droit est :" + lo*la);
    alert("l'aire latérale du pave droit est :"+ 2*h*(lo+la));
    alert("l'aire totale du pave droit est :"+ (2*h*(lo+la)+2*lo*la));
    alert("le volume du pave droit est :" + lo*la*h);

}



var cylindre=function(r,h){
    alert("le perimètre de base cylindre est " + (22/7)*2*r);
    alert("l'aire de base cylibndre est :" + (22/7)*r*r);
    alert("l'aire latérale cylindre est :"+ (22/7)*2*r*h);
    alert("l'aire totale cylindre est :"+ (22/7)*r*r*2+(22/7)*2*r*h);
    alert("le volume cylindre est :" + (22/7)*r*r*h);

}


// Code concernant les vecteurs et autres

// pdt est point dans un espace de dimension trois

var remplirpdt=function(pdt){
    var x1,y1,z1;
    x1=parseFloat(prompt("Entrer le x"));
    y1=parseFloat(prompt("Entrer le y"));
    z1=parseFloat(prompt("Entrer le z"));
     pdt={
        x: x1,
        y: y1,
        z: z1,
    }
}

// var vecteurpdt=function( pdt1, pdt2){
   // var pdt={

  //  }}


var equdroitedeuxplan=function(x1,y1,x2,y2){
    var a,b;
    a =parseFloat ((y1-y2)/(x1-x2));
    b = parseFloat(y2-a*x2);
    alert("l/'equation de droite est : "+ a +" x + " + "(-y) + " + b +"= 0");
    
}
//var equplan=function(x1,y1,z1,x2,y2,z2,x3,y3,z3){



// Statistique


//   Remplissage
var mod=[];
var n=parseFloat(prompt("Etrer le nombre de colonne de votre tableau"));
for (var i=0;i<n ;i++ ){
    mod.push(parseFloat(prompt("Entrer la modalité en position :"+(i+1))));
}

for (var i=0; i<n;i++){
    document.write("la modalité à la position "+(i+1) +"est : "+ mod[i+1]);
}

var eff=[];
for (var i=0;i<n ;i++ ){
    mod.push(parseFloat(prompt("Entrer l'effectif de :"+ (i+1))));
}

for (var i=0; i<n;i++){
    alert("l'effectif à la position "+(i+1) +"est : "+ mod[i+1]);
}

// calcul
var effcum=[];
for (var i=0;i<n ;i++ ){
    
}

for (var i=0; i<n;i++){
    alert("la modalité à la position "+i +"est : "+ mod[i]);
}






























    //alert("Calcul de pgcd") ;
        var a,b,x1,x2,y1,y2,z1,z2,z3;
        a=parseInt(prompt(" Enter a"));
        b=parseInt(prompt("entrer b"));
      //  alert("pgcd=  " +pgcdRec(a,b) + "  et ppcm est "  +ppcmRec(a,b));
     x1=parseFloat(prompt(" Enter x1"));
     y1=parseFloat(prompt("entrer y1"));
     x2=parseFloat(prompt(" Enter x2"));
       // y2=parseFloat(prompt("entrer y2"));
       // equdroitedeuxplan(x1,y1,x2,y2);
       // cube (a);
       // pave_droit(x1,y1,x2);
       // cylindre(a,b);