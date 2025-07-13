
/**
 * Erzeugt das Sprachmodell (LSTM) für Next Word Prediction
 *
 * @param {number} vocabSize - Größe des Vokabulars
 * @param {number} maxLen - Maximale Eingabesequenzlänge
 * @param {number} embeddingDim - Dimension der Embeddings
 * @param {number} lstmUnits - Anzahl der LSTM-Einheiten
 * @returns {tf.LayersModel} Das erstellte Modell
 */
export function createLanguageModel(vocabSize, maxLen, embeddingDim, lstmUnits) {
    const model = tf.sequential();

    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embeddingDim,
        inputLength: maxLen,
        maskZero: true // wichtig bei Padding!
    }));

    // Erste LSTM-Schicht: gibt Sequenzen weiter
    model.add(tf.layers.lstm({
        units: lstmUnits,
        returnSequences: true,
        dropout: 0.2,           // Dropout auf Inputs der LSTM-Zellen
        recurrentDropout: 0.2,  // Dropout auf rekurrente Verbindungen
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
    }));

    // Zweite LSTM-Schicht: gibt Vektor aus
    model.add(tf.layers.lstm({
        units: lstmUnits,
        returnSequences: false,
        dropout: 0.2,           // Dropout auf Inputs der LSTM-Zellen
        recurrentDropout: 0.2,  // Dropout auf rekurrente Verbindungens
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
    }));

    model.add(tf.layers.dense({
        units: vocabSize,
        activation: 'softmax',
    }));

    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}
