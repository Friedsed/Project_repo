function calculate() {
    const numElements = parseInt(document.getElementById("numElements").value);
    const elementInputs = document.getElementById("elementInputs");
    const cis = [];

    for (let i = 1; i <= numElements; i++) {
        const input = parseFloat(document.getElementById(`element${i}`).value);
        cis.push(input);
    }
    document.getElementById("result").innerHTML = "Multiplied Array: " + cis.join(", ");
   
  
}



function calculates() {
    const numElements = parseInt(document.getElementById("numElements").value);
    const eI = document.getElementById("eI");
    const nis = [];

    for (let i = 1; i <= numElements; i++) {
        const input = parseFloat(document.getElementById(`element0${i}`).value);
        nis.push(input);
    }

    const multipliedArray0= nis.map(element => element * 2);

    document.getElementById("results").innerHTML = "Multiplied Array: " + nis.join(", ");
   
  
}


