// sentence-grouper.js

export class SentenceGrouper {
    constructor() {
        this.abbreviations = ['i.e.', 'e.g.', 'Mr.', 'Mrs.', 'Dr.', 'etc.'];
    }

    isSentenceEnd(token) {
        if (!token) return false;
        if (token.endsWith('.')) {
            if (token.endsWith('.)') || token.endsWith('.]') || token.endsWith('.}')) return true;
            if (this.abbreviations.includes(token)) return false;
            return true;
        }
        return false;
    }

    group(data) {
        const grouped = [];
        let current = [];

        for (let item of data) {
            const word = item.token;
            if (word) current.push(item);
            if (this.isSentenceEnd(word)) {
                grouped.push(current);
                current = [];
            }
        }

        if (current.length > 0) grouped.push(current);

        return grouped;
    }
}

// Exportiere auch die reine Token-Funktion
export function groupTokensIntoSentences(tokenList) {
    const abbreviations = ['i.e.', 'e.g.', 'Mr.', 'Mrs.', 'Dr.', 'etc.'];

    function isSentenceEnd(token) {
        if (!token) return false;
        if (token.endsWith('.')) {
            if (token.endsWith('.)') || token.endsWith('.]') || token.endsWith('.}')) return true;
            if (abbreviations.includes(token)) return false;
            return true;
        }
        return false;
    }

    const grouped = [];
    let current = [];

    for (let token of tokenList) {
        if (token) current.push(token);
        if (isSentenceEnd(token)) {
            grouped.push(current);
            current = [];
        }
    }

    if (current.length > 0) grouped.push(current);
    return grouped;
}
