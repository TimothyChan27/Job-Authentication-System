document.addEventListener('DOMContentLoaded', () => {
    const detectButton = document.getElementById('detect-button');
    const resultElement = document.getElementById('result');

    detectButton.addEventListener('click', async () => {
        const jobPostingText = document.getElementById('job-posting-text').value;

        // Send the data to the backend
        try {
            const response = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: jobPostingText }),
            });

            const result = await response.json();

            // Update the result element based on the response
            if (result.isReal) {
                resultElement.textContent = 'This Job Posting is Real';
                resultElement.style.color = '#2D55CC'; // Dark blue color
            } else {
                resultElement.textContent = 'This Job Posting is Fake';
                resultElement.style.color = '#FF4041'; // Red color
            }
        } catch (error) {
            console.error('Error detecting job posting:', error);
            resultElement.textContent = 'Error processing request';
            resultElement.style.color = 'black'; // Default color for error
        }
    });
});
