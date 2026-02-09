

var password=prompt("Enter the password");
while(password!='Sedjro'){
    var password=prompt("Entrer the password");

}





function pgcd(a,b){
    if(a%b===0){
        return b;
    }
    return pgcd(a,a%b);
}
   



    alert('pgcd='+pgcd(3,8));



    var name=prompt('Enter your name please');
    var text='Welcome my friend ', resultat;
    resultat=text + name;
    alert(resultat);
    alert('Now we will do some exerxices let \'s calculate the aire of the cube');
    var a,Ab,Al,At,Pb,V ;
    alert("Le cube est un solide geometrique. Il a 6 faces superposables ;8 sommets et  12 arrets de meme longueur. Un cube est pave droit particulier\n\n");
    alert(" En prenant pour  , a l arrete ; on a \n quelques formules Ab = a*a;Pb =4*a;Al =Pb*a ;At = 6*a*a;V = a*a*a  ");
    alert(" Ce code trouve les aires et les volume du cubes ");
    a=parseInt(prompt(" Entrer l arrete du cube \n\n"));
    Ab = a*a;
    Pb =4*a;
    Al =Pb*a ;
    At = 6*a*a;
    V = a*a*a ;
    alert("the base area equal to "+Ab);
                    
    alert("the base perimeter equal "+Pb);
                    
    alert("the lateral area equal to "+Al);
                    
    alert("the total area equal to "+At);
                    
    alert("the volume equal to "+V);

    
        