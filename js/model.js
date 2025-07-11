
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
    }));

    // Erste LSTM-Schicht: gibt Sequenzen weiter
    model.add(tf.layers.lstm({
        units: lstmUnits,
        returnSequences: true,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
    }));

    // Zweite LSTM-Schicht: gibt Vektor aus
    model.add(tf.layers.lstm({
        units: lstmUnits,
        returnSequences: false,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
    }));

    model.add(tf.layers.dense({
        units: vocabSize,
        activation: 'softmax',
    }));

    model.compile({
        optimizer: tf.train.adam(0.0005),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}
