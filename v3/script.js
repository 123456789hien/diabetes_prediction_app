let model;
let trainingData;

async function loadData() {
    const response = await fetch('data/diabetes_raw_cleaned_25k.csv');
    const data = await response.text();
    const parsedData = Papa.parse(data, { header: true, dynamicTyping: true }).data;
    trainingData = parsedData;
    console.log('Data loaded successfully');
}

async function trainModel() {
    await loadData();
    
    const inputData = [];
    const labels = [];

    // Tiền xử lý dữ liệu
    trainingData.forEach(row => {
        // Các bước tiền xử lý và chuẩn hóa
        const features = [
            row.age / 100,       // Normalize age
            row.bmi / 40,        // Normalize BMI
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

    // Chuyển đổi thành TensorFlow.js tensor
    const xs = tf.tensor2d(inputData);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    // Xây dựng mô hình neural network
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [xs.shape[1]], units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: 32, activation: 'relu'}));  // Layer thứ 2
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'})); // Output layer

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Huấn luyện mô hình
    await model.fit(xs, ys, {
        epochs: 10,         // Số lần huấn luyện qua toàn bộ dữ liệu
        batchSize: 32,      // Kích thước mỗi batch
        validationSplit: 0.2 // Dùng 20% dữ liệu cho kiểm tra
    });

    // Lưu mô hình vào model.json
    await model.save('localstorage://diabetes-model');

    alert('Model trained and saved in local storage!');
    console.log('Model trained');
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
