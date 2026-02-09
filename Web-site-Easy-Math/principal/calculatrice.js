let displayValue = "";

function appendToDisplay(value) {
    displayValue += value;
    document.getElementById('display').value = displayValue;
}

function clearDisplay() {
    displayValue = "";
    document.getElementById('display').value = "";
}

function calculate() {
    try {
        displayValue = eval(displayValue).toString();
        document.getElementById('display').value = displayValue;
    } catch (error) {
        document.getElementById('display').value = "Error";
    }
}

function calculateTrig(func) {
    try {
        if (func === 'sin') {
            displayValue = Math.sin(eval(displayValue)).toString();
        } else if (func === 'cos') {
            displayValue = Math.cos(eval(displayValue)).toString();
        } else if (func === 'tan') {
            displayValue = Math.tan(eval(displayValue)).toString();
        }
        document.getElementById('display').value = displayValue;
    } catch (error) {
        document.getElementById('display').value = "Error";
    }
}

function calculateMath(func) {
    try {
        if (func === 'sqrt') {
            displayValue = Math.sqrt(eval(displayValue)).toString();
        } else if (func === 'log') {
            displayValue = Math.log(eval(displayValue)).toString();
        } else if (func === 'exp') {
            displayValue = Math.exp(eval(displayValue)).toString();
        }
        document.getElementById('display').value = displayValue;
    } catch (error) {
        document.getElementById('display').value = "Error";
    }
}
