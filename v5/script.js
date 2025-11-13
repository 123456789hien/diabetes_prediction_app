let model;
let history = { loss: [], accuracy: [] };  // Lưu lại thông tin về loss và accuracy trong quá trình huấn luyện

async function loadData() {
    const response = await fetch('data/diabetes_raw_cleaned_25k.csv');
    const data = await response.text();
    const parsedData = Papa.parse(data, { header: true, dynamicTyping: true }).data;
    return parsedData;
}

async function trainModel() {
    document.getElementById("trainingStatus").classList.remove("hidden");

    const trainingData = await loadData();
    
    const inputData = [];
    const labels = [];

    // Tiền xử lý dữ liệu
    trainingData.forEach(row => {
        const features = [
            row.age / 100,        // Normalize age
            row.bmi / 40,         // Normalize BMI
            row.HbA1c_level / 10, // Normalize HbA1c
            row.blood_glucose_level / 200, // Normalize blood glucose
            row.hypertension,
            row.heart_disease,
            row.smoking_history === "current" ? 1 : 0,
            row.smoking_history === "former" ? 1 : 0,
            row.smoking_history === "ever" ? 1 : 0,
            row.smoking_history === "not current" ? 1 : 0,
            row.gender === "Male" ? 1 : 0
        ];

        const label = row.diabetes;  // 0 or 1 for diabetes

        inputData.push(features);
        labels.push(label);
    });

    const xs = tf.tensor2d(inputData);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [xs.shape[1]], units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Huấn luyện mô hình và lưu kết quả vào history
    const result = await model.fit(xs, ys, {
        epochs: 10,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                history.loss.push(logs.loss);
                history.accuracy.push(logs.acc);
                updateTrainingChart();  // Cập nhật biểu đồ huấn luyện sau mỗi epoch
            }
        }
    });

    await model.save('localstorage://diabetes-model');
    alert('Model trained and saved in local storage!');
    console.log('Model trained');

    document.getElementById("trainingStatus").classList.add("hidden");
}

function predict(input) {
    const inputTensor = tf.tensor2d([input]);
    const prediction = model.predict(inputTensor);
    return prediction.dataSync()[0]; // Trả về xác suất
}

document.getElementById("predictForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const input = [
        parseFloat(document.getElementById("age").value),
        parseFloat(document.getElementById("bmi").value),
        parseFloat(document.getElementById("HbA1c_level").value),
        parseFloat(document.getElementById("blood_glucose_level").value),
        parseInt(document.getElementById("hypertension").value),
        parseInt(document.getElementById("heart_disease").value),
        document.getElementById("smoking_history").value === "current" ? 1 : 0,
        document.getElementById("gender").value === "Male" ? 1 : 0
    ];

    const risk = predict(input);
    const label = risk >= 0.5 ? "Positive" : "Negative";
    const probability = (risk * 100).toFixed(1);

    document.getElementById("result").textContent = `Diabetes Risk: ${label}`;
    document.getElementById("probability").textContent = `Probability: ${probability}%`;
    document.getElementById("comment").textContent = label === "Positive" ? "Consult a doctor." : "Low risk.";

    document.getElementById("output").classList.remove("hidden");
});

document.getElementById("trainButton").addEventListener("click", async function(event) {
    await trainModel();
});

// Cập nhật biểu đồ huấn luyện
function updateTrainingChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    if (window.trainingChart) {
        window.trainingChart.update();
    } else {
        window.trainingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from(Array(history.loss.length).keys()),  // epoch number
                datasets: [
                    {
                        label: 'Loss',
                        data: history.loss,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        fill: false
                    },
                    {
                        label: 'Accuracy',
                        data: history.accuracy,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        beginAtZero: true
                    },
                    y: {
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }
}
