
import { LMPreprocessor } from './lm-preprocessing.js'; // neue Klasse importieren
import { createLanguageModel } from './model.js';

export async function trainLanguageModel({
                                             tokenGroups,
                                             valTokenGroups, // Dev-/Testdaten, erwartet {X, y} als Arrays
                                             vocab,
                                             maxLen,
                                             embeddingDim,
                                             lstmUnits,
                                             epochs,
                                             batchSize,
                                         }) {
    console.log("📦 Training mit Parametern:", { maxLen, embeddingDim, lstmUnits, epochs, batchSize });

    // Preprocessor initialisieren
    const preprocessor = new LMPreprocessor(vocab, maxLen);
console.log(tokenGroups);
    // Trainingsdaten vorverarbeiten
    const { X, y } = preprocessor.preprocessData(tokenGroups);

    // Analyse-Logs zum Training
    const UNK_ID = vocab['<UNK>'];
    let totalUnks = 0;
    let totalTokens = 0;

    X.forEach(seq => {
        totalUnks += seq.filter(id => id === UNK_ID).length;
        totalTokens += seq.length;
    });

    const unkRatio = (totalUnks / totalTokens) * 100;
    console.log(`📊 UNK-Anteil in Inputs: ${unkRatio.toFixed(2)} %`);

    const uniqueTargets = new Set(y);
    console.log(`📊 Anzahl einzigartiger Zielwörter: ${uniqueTargets.size} von ${Object.keys(vocab).length}`);
    console.log(`📊 Gesamtzahl Trainingsbeispiele: ${X.length}`);

    console.log('X:', X);
    console.log('X[0]:', X[0]);
    console.log('X ist Array?', Array.isArray(X));
    console.log('X[0] ist Array?', Array.isArray(X[0]));

    // Tensoren erstellen
    const X_padded = tf.tensor2d(X);
    const yTensor = tf.tensor1d(y, 'int32');

    // 🧪 Validation-Preprocessing
    const { X: valX, y: valY } = preprocessor.preprocessData(valTokenGroups);

    // Modell erstellen
    const model = createLanguageModel(Object.keys(vocab).length, maxLen, embeddingDim, lstmUnits);
    model.summary();

    console.log("X_padded shape:", X_padded.shape);
    const id2word = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));

    // 5 zufällige Trainingsbeispiele loggen
    for (let i = 0; i < 5; i++) {
        const seqIndex = Math.floor(Math.random() * X.length);
        const inputTokens = X[seqIndex];
        const targetTokenId = y[seqIndex];

        const inputWords = inputTokens.map(id => id2word[id] || '<UNK>');
        const targetWord = id2word[targetTokenId] || '<UNK>';

        console.log(`📚 Beispiel ${i + 1}`);
        console.log("📥 Input:", inputWords.join(' '));
        console.log("🎯 Target:", targetWord);
    }

    const metrics = [];

    // Modell trainieren
    await model.fit(X_padded, yTensor, {
        epochs,
        batchSize,
        shuffle: true,
        validationData: [
            tf.tensor2d(valX),
            tf.tensor1d(valY, 'int32')
        ],
        callbacks: [
            {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`📈 Epoch ${epoch + 1}:`);
                    console.log(`   🔹 Training Loss = ${logs.loss.toFixed(4)}`);
                    if (logs.val_loss !== undefined) {
                        console.log(`   🔸 Validation Loss = ${logs.val_loss.toFixed(4)}`);
                    }

                    metrics.push({
                        epoch,
                        loss: logs.loss,
                        val_loss: logs.val_loss,
                    });

                    tfvis.render.linechart(
                        { name: '📉 Training vs. Validation Loss' },
                        {
                            values: [
                                metrics.map(m => ({ x: m.epoch + 1, y: m.loss })),
                                metrics.map(m => ({ x: m.epoch + 1, y: m.val_loss }))
                            ],
                            series: ['Training Loss', 'Validation Loss']
                        },
                        {
                            xLabel: 'Epoche',
                            yLabel: 'Loss',
                            width: 500,
                            height: 300
                        }
                    );

                    if ((epoch + 1) % 2 === 0) {
                        await model.save('indexeddb://checkpoint');
                        console.log(`💾 Modell-Checkpoint gespeichert (Epoch ${epoch + 1})`);
                    }
                }
            },
            new EarlyStopping(3)
        ]
    });

    await tf.nextFrame();

    await model.save('downloads://trained-lm');
    console.log("✅ Modell gespeichert");

    return model;
}

class EarlyStopping extends tf.Callback {
    constructor(patience) {
        super();
        this.patience = patience;
        this.bestLoss = Infinity;
        this.wait = 0;
    }

    async onEpochEnd(epoch, logs) {
        const currentLoss = logs.val_loss;
        if (currentLoss < this.bestLoss - 1e-2) {
            this.bestLoss = currentLoss;
            this.wait = 0;
        } else {
            this.wait++;
            if (this.wait >= this.patience) {
                console.log(`⛔ Training stopped early at epoch ${epoch + 1}`);
                this.model.stopTraining = true;
            }
        }
    }
}
