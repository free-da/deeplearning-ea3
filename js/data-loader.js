export class DataLoader {
    constructor() {
        this.trainPath = "data/train.csv";
        this.devPath = "data/dev.csv";
        this.testPath = "data/test.csv";
    }

    async loadCSV(path) {
        const response = await fetch(path);
        const csvText = await response.text();
        // CSV in Zeilen splitten und erste Zeile (Header) überspringen
        const lines = csvText.trim().split("\n").slice(1);

        const data = [];

        for (let line of lines) {
            const [token, label] = line.split(",");
            if (token && label) {
                data.push({ token, label });
            }
        }

        return data;
    }

    async loadAll() {
        const trainData = await this.loadCSV(this.trainPath);
        const devData = await this.loadCSV(this.devPath);
        const testData = await this.loadCSV(this.testPath);
        return { trainData, devData, testData };
    }
}
