async function fetchDiseaseInfo() {
    const name = document.getElementById('nameInput').value;
    const diseaseName = document.getElementById('diseaseInput').value;
    const resultContainer = document.getElementById('result');
    const downloadButton = document.getElementById('downloadButton');
    resultContainer.innerHTML = '';
    downloadButton.style.display = 'none';

    if (!name || !diseaseName) {
        alert('Please enter your name and a disease name');
        return;
    }

    try {
        const response = await fetch('/get_disease', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, disease: diseaseName })
        });

        if (response.ok) {
            const data = await response.json();
            resultContainer.innerHTML = `
                <div class="result-card">
                    <h2>Medical Report for ${data.Name}</h2>
                    <p><strong>Disease:</strong> ${data.Disease}</p>
                    <p><strong>Information:</strong> ${data.Information}</p>
                    <p><strong>Symptoms:</strong> ${data.Symptoms}</p>
                    <p><strong>Treatment:</strong> ${data.Treatment}</p>
                    <p><strong>Precaution:</strong> ${data.Precaution}</p>
                </div>
            `;
            downloadButton.style.display = 'inline-block';
        } else {
            const error = await response.json();
            resultContainer.innerHTML = `<p class="error">${error.error}</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultContainer.innerHTML = `<p class="error">Failed to fetch data</p>`;
    }
}

async function downloadReport() {
    const name = document.getElementById('nameInput').value;
    const diseaseName = document.getElementById('diseaseInput').value;

    try {
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, disease: diseaseName })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${name}_${diseaseName}_report.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const error = await response.json();
            alert(error.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate report');
    }
}
