function getScoreLabel(score, passingRequirement) {
    if (score < passingRequirement) {
        return "Weak";
    } else if (score < 75) { // Assuming 75 is the threshold for "normal" if it's not "Weak"
        return "Normal";
    } else {
        return "Strong";
    }
}

document.getElementById('assessmentForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent default form submission

    const additionScore = parseInt(document.getElementById('additionScore').value);
    const subtractionScore = parseInt(document.getElementById('subtractionScore').value);
    const multiplicationScore = parseInt(document.getElementById('multiplicationScore').value);
    const divisionScore = parseInt(document.getElementById('divisionScore').value);
    const passingScore = parseInt(document.getElementById('passingScore').value);


    // Basic validation
    if (isNaN(additionScore) || isNaN(subtractionScore) || isNaN(multiplicationScore) || isNaN(divisionScore) || isNaN(passingScore)) {
        alert('Please fill in all fields with valid numbers.');
        return;
    }

        // Apply the score labeling function to each score
    const additionLabel = getScoreLabel(additionScore, passingScore);
    const subtractionLabel = getScoreLabel(subtractionScore, passingScore);
    const multiplicationLabel = getScoreLabel(multiplicationScore, passingScore);
    const divisionLabel = getScoreLabel(divisionScore, passingScore);


    const data = {
        transcript: {
            addition: additionScore,
            subtraction: subtractionScore,
            multiplication: multiplicationScore,
            division: divisionScore
        },
        // Include the labels in the data sent to the backend
        scoreLabels: {
            addition: additionLabel,
            subtraction: subtractionLabel,
            multiplication: multiplicationLabel,
            division: divisionLabel
        },
        passingRequirement: passingScore
    };

    const responseBlock = document.getElementById('responseBlock');
    responseBlock.textContent = 'Submitting data to backend...';
    responseBlock.style.color = '#333'; // Reset color in case of previous error

    try {
        const response = await fetch('http://127.0.0.1:5000/assess', { // Replace with your backend URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const result = await response.json();
        responseBlock.textContent = JSON.stringify(result, null, 2); // Pretty print JSON
        responseBlock.style.color = '#333'; // Ensure text color is normal for success
    } catch (error) {
        console.error('Error:', error);
        responseBlock.textContent = `Error: ${error.message}. Please ensure the backend is running.`;
        responseBlock.style.color = 'red';
    }
});