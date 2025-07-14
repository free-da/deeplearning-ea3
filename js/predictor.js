
import { padSequences, tokenizeText } from './tokenizer.js';

/**
 * Wrapper-Klasse für Vorhersagen
 */
export class Predictor {
    constructor(model, vocab, maxLen) {
        this.model = model;
        this.vocab = vocab;
        this.id2word = Object.entries(vocab).reduce((obj, [word, id]) => {
            obj[id] = word;
            return obj;
        }, {});
        this.maxLen = maxLen;
    }

    setModel(model, vocab, maxLen) {
        this.model = model;
        this.vocab = vocab;
        this.id2word = Object.entries(vocab).reduce((obj, [word, id]) => {
            obj[id] = word;
            return obj;
        }, {});
        this.maxLen = maxLen;
    }

    async predict(inputText, topK = 5) {
        const tokens = tokenizeText(inputText);
        const tokenIds = tokens.map(t => this.vocab[t] ?? this.vocab['<UNK>']); // Fallback
        const padded = padSequences([tokenIds], this.maxLen); // [1, maxLen]

        const input = tf.tensor2d(padded);
        console.log("Padded input shape:", padded[0].length);
        console.log("Padded tokens:", padded[0].map(id => this.id2word[id]));
        const prediction = this.model.predict(input);
        const probs = await prediction.data(); // statt .dataSync()
        await tf.nextFrame(); // Lässt den Browser Rendering/Events verarbeiten

        // Top-K Wahrscheinlichkeiten
        const sorted = [...probs.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, topK)
            .map(([id, prob]) => ({
                word: this.id2word[id] || '[UNK]',
                prob,
            }));
        console.log("Input tokens:", tokens);
        console.log("Token IDs:", tokenIds);
        console.log("Top IDs:", sorted.map(p => p.word));

        tf.dispose([input, prediction]);
        return sorted;
    }
    async predictNextWord(inputText) {
        const predictions = await this.predict(inputText, 1);
        console.log(predictions);
        return predictions.length > 0 ? predictions[0].word : null;
    }

}
