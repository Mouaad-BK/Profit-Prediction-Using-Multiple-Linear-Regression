document.getElementById('pcaCheckbox').addEventListener('change', function() {
    const pcaOptions = document.getElementById('pcaOptions');
    const btnAppACP = document.getElementById('btnAppACP');
    
    // Afficher ou masquer les options PCA
    pcaOptions.style.display = this.checked ? 'block' : 'none';
    
    // Afficher ou masquer le bouton en fonction de l'état de la case à cocher
    btnAppACP.style.display = this.checked ? 'inline-block' : 'none';
});

document.getElementById('trainSplit').addEventListener('input', function() {
    document.getElementById('trainValue').textContent = `${this.value}%`;
});

function trainModel() {
    console.log("Entraînement du modèle...");
}

function testModel() {
    console.log("Test du modèle...");
}

document.getElementById("pcaCheckbox").addEventListener("change", function() {
    document.getElementById("pcaOptions").style.display = this.checked ? "block" : "none";
});

function updateTrainValue() {
    const trainSplit = document.getElementById('trainSplit');
    const trainValue = document.getElementById('trainValue');
    trainValue.textContent = trainSplit.value + '%';
}
