import { SentenceGrouper } from './sentence-grouper.js';

export class LMPreprocessor {
    constructor(vocab, maxLen, minRealTokens = 8) {
        this.vocab = vocab;
        this.maxLen = maxLen;
        this.minRealTokens = minRealTokens;

        this.UNK_ID = vocab['<UNK>'];
        this.PAD_ID = vocab['<PAD>'];

        this.grouper = new SentenceGrouper();
        console.log('Vocab:', this.vocab);
        console.log('UNK_ID:', this.UNK_ID);
        console.log('PAD_ID:', this.PAD_ID);

    }

    /**
     * Entfernt Zitierungen aus gruppierten Token-Sätzen.
     * @param {Array[]} tokenGroups - Array von Token-Arrays (Sätze)
     * @returns {Array[]} Bereinigte Token-Gruppen
     */
    removeCitations(tokenGroups) {
        const citationRegex = /\[\s*(\d+([–,-]\d+)?(\s*,\s*\d+([–,-]\d+)?)*)\s*\]/g;
        return tokenGroups.map(tokens => {
            const sentence = tokens.join(' ');
            const cleaned = sentence.replace(citationRegex, '').trim();
            const cleanedTokens = cleaned.split(/\s+/).filter(t => t.length > 0);
            return cleanedTokens;
        }).filter(group => group.length > 0);
    }

    /**
     * Wandelt Token-Gruppen (Array von Array von Tokens) in ID-Sequenzen mit Padding um.
     * @param {Array[]} tokenGroups
     * @returns {number[][]} gepaddete ID-Sequenzen
     */
    tokensToPaddedSequences(tokenGroups) {
        // 1. Zitierungen entfernen
        const cleanedGroups = this.removeCitations(tokenGroups);

        // 2. Tokens zu IDs
        const sequences = cleanedGroups.map(tokens =>
            tokens.map(t => this.vocab[t] !== undefined ? this.vocab[t] : this.UNK_ID)
        );
        // 3. Padding
        return sequences.map(seq => this.padSequenceToMaxLen(seq));
    }

    /**
     * Pad oder trimmt Sequenz auf maxLen mit PAD_ID links
     * @param {number[]} seq
     * @returns {number[]}
     */
    padSequenceToMaxLen(seq) {
        if (seq.length > this.maxLen) {
            return seq.slice(seq.length - this.maxLen);
        } else if (seq.length < this.maxLen) {
            const padArray = new Array(this.maxLen - seq.length).fill(this.PAD_ID);
            return padArray.concat(seq);
        } else {
            return seq;
        }
    }

    /**
     * Für Trainingsdaten: Erzeugt (X,y) Paare für Next-Word-Prediction.
     * Filtert Ziele <PAD> oder <UNK>, mindestanzahl realer Tokens.
     * @param {Array[]} sequences - gepaddete ID-Sequenzen
     * @returns {{X: number[][], y: number[]}}
     */
    prepareTrainingData(sequences) {
        const X = [];
        const y = [];

        let total = 0;
        console.log('Sequences in prepareTrainingData:', sequences);

        for (const seq of sequences) {
            if (!seq) {
                console.warn('Found undefined sequence!');
                continue;
            }
            if (!Array.isArray(seq)) {
                console.warn('Found non-array sequence:', seq);
                continue;
            }
            for (let i = 1; i < seq.length; i++) {
                const target = seq[i];
                if (target === this.UNK_ID || target === this.PAD_ID) continue;

                total++;

                const startIdx = Math.max(0, i - this.maxLen);
                const inputSeq = seq.slice(startIdx, i);
                const paddedSeq = this.padSequenceToMaxLen(inputSeq);

                const realTokenCount = paddedSeq.filter(id => id !== this.PAD_ID).length;
                if (realTokenCount < this.minRealTokens) continue;

                X.push(paddedSeq);
                y.push(target);
            }
        }
        console.log(`Gesamt: ${total} | Verwendet: ${X.length} | Verworfene: ${total - X.length}`);

        return { X, y };
    }

    /**
     * Vorverarbeitung von Trainingsdaten (Token-Gruppen von Objekten mit .token)
     * @param {Array[]} tokenGroups - Array von Sätzen mit {token: string}
     * @returns {{X: number[][], y: number[]}}
     */
    preprocessData(tokenGroups) {
        // tokenGroups: Array von Sätzen, jeder Satz ist Array von { token: string }
        // 1. Gruppierung direkt auf flachem Array von Token-Objekten
        const flatTokenObjects = tokenGroups.flat();

        // 2. Von gruppierten Token-Objekten zu Arrays von Token-Strings
        const groupedTokens = this.grouper.group(flatTokenObjects);

        // 3. Jetzt Padding + Umwandlung in IDs + Zitations-Entfernung in tokensToPaddedSequences
        const paddedSequences = this.tokensToPaddedSequences(groupedTokens);

        // 4. (X,y) erzeugen
        return this.prepareTrainingData(paddedSequences);
    }

}
