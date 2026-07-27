import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const outputDir = path.resolve("..", "workbooks");
const previewDir = path.resolve("..", "previews");
const opporSource = String.raw`C:\Users\westl\PycharmProjects\pythonProject\Data\oppor.xlsx`;
const factsetSource = String.raw`C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\Factset_re_study.xlsx`;

const factsetSheets = [
  "S&P500_FwdEPS",
  "S&P500_FwdEPS(1Y)",
  "S&P500_FwdSales(1Y)",
  "S&P500 FwdOPMargin(1Y)",
  "S&P500 Fwd_OpCashflow(1Y)",
  "S&P500 FwdEV_EBITDA(1Y)",
  "S&P500 Fwd_FreeCashflow(1Y)",
  "S&P500_earning_surprise",
  "sales_surprise",
  "C_RPO",
  "Total_RPO",
  "TG_Price",
  "EPS_Revision",
  "Sales_Revision",
];

function columnName(columnNumber) {
  let value = columnNumber;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

async function loadWorkbook(filePath) {
  return SpreadsheetFile.importXlsx(await FileBlob.load(filePath));
}

async function saveWorkbook(workbook, filePath) {
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(filePath);
}

async function editOppor() {
  const workbook = await loadWorkbook(opporSource);
  const sheet = workbook.worksheets.getItem("tickers");
  const target = sheet.getRange("EN1");
  const before = target.values[0][0];
  if (before !== "ROG SW Equity") {
    throw new Error(`Expected tickers!EN1 to contain ROG SW Equity, got ${before}`);
  }
  target.values = [["SAN FP Equity"]];

  const outputPath = path.join(outputDir, "oppor_SAN.xlsx");
  await saveWorkbook(workbook, outputPath);
  const preview = await workbook.render({
    sheetName: "tickers",
    range: "EH1:ET2",
    scale: 2,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewDir, "oppor_tickers.png"),
    new Uint8Array(await preview.arrayBuffer()),
  );
  const inspected = await workbook.inspect({
    kind: "table",
    range: "tickers!EH1:ET2",
    include: "values,formulas",
    tableMaxRows: 2,
    tableMaxCols: 13,
  });
  return { outputPath, before, after: target.values[0][0], inspect: inspected.ndjson };
}

async function editFactset() {
  const workbook = await loadWorkbook(factsetSource);
  const placements = [];
  for (const sheetName of factsetSheets) {
    const sheet = workbook.worksheets.getItem(sheetName);
    const header = sheet.getRange("A2:HZ2").values[0];
    const index = header.findIndex((value) => value === "ROG-CH^");
    if (index < 0) {
      throw new Error(`${sheetName}: ROG-CH^ header not found`);
    }
    const address = `${columnName(index + 1)}2`;
    sheet.getRange(address).values = [["SAN-FR^"]];
    placements.push({ sheetName, address });
  }

  const outputPath = path.join(outputDir, "Factset_re_study_SAN.xlsx");
  await saveWorkbook(workbook, outputPath);
  for (const placement of placements) {
    const columnNumber = (() => {
      let number = 0;
      for (const char of placement.address.replace(/\d/g, "")) {
        number = number * 26 + char.charCodeAt(0) - 64;
      }
      return number;
    })();
    const start = columnName(Math.max(1, columnNumber - 2));
    const end = columnName(columnNumber + 2);
    const preview = await workbook.render({
      sheetName: placement.sheetName,
      range: `${start}1:${end}4`,
      scale: 2,
      format: "png",
    });
    const safeName = placement.sheetName.replace(/[^A-Za-z0-9_-]+/g, "_");
    await fs.writeFile(
      path.join(previewDir, `factset_${safeName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
  const errorScan = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan",
  });
  return { outputPath, placements, errorScan: errorScan.ndjson };
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const mode = process.argv[2] ?? "all";
if (mode === "oppor") {
  console.log(JSON.stringify({ oppor: await editOppor() }, null, 2));
} else if (mode === "factset") {
  console.log(JSON.stringify({ factset: await editFactset() }, null, 2));
} else if (mode === "all") {
  const oppor = await editOppor();
  const factset = await editFactset();
  console.log(JSON.stringify({ oppor, factset }, null, 2));
} else {
  throw new Error(`Unknown mode: ${mode}`);
}
