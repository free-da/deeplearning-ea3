export function prepareLanguageModelData(sequences, maxLen) {
    const X = [];
    const y = [];

    for (const seq of sequences) {
        for (let i = 1; i < seq.length; i++) {
            // Hole bis zu maxLen Tokens vor dem Zieltoken
            const startIdx = Math.max(0, i - maxLen);
            const inputSeq = seq.slice(startIdx, i); // Eingabe mit max maxLen Tokens
            const paddedSeq = padSequenceToMaxLen(inputSeq, maxLen);

            X.push(paddedSeq);
            y.push(seq[i]);
        }
    }

    return { X, y };
}

function padSequenceToMaxLen(seq, maxLen, padValue = 0) {
    if (seq.length > maxLen) {
        return seq.slice(seq.length - maxLen); // auf maxLen kürzen
    } else if (seq.length < maxLen) {
        // links mit Pad auffüllen
        const padArray = new Array(maxLen - seq.length).fill(padValue);
        return padArray.concat(seq);
    } else {
        return seq;
    }
}
