

function changeCouleur(mon_rectangle){
    var text_a_changer= document.getElementById("texte_a_modifier");

    var style=window.getComputedStyle(mon_rectangle);

    var couleur_a_appliquer=style.getPropertyValue('background-color');

    texte_a_changer.style.color = couleur_a_appliquer;
}




function bonjour(){

    var prenom=document.getElementById("prenom").value;

    var nom=document.getElementById("nom").value;

    var texte= "Bonjour "+prenom+" ! ";
    document.getElementById("salutations").innerHTML=texte;
}