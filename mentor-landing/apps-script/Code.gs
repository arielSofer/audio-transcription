// Google Apps Script – מחובר ל-Google Sheet פרטי.
// כל הרשמה: נשמרת כשורה בגיליון + נשלח לך מייל עדכון.

const NOTIFY_EMAIL = 'arielvdcr@gmail.com'; // לשנות אם רוצים מייל אחר
const SHEET_ID = '1w9C3RFT2SLzKdK0794GD2ANGlmKRP6ANid0C7dE4bxw'; // הגיליון הפרטי שלך (כבר נוצר)

const HEADERS = [
  'תאריך', 'שם', 'טלפון', 'אימייל', 'ניסיון בשוק', 'מצב נוכחי', 'ניסיונות שכשלו',
  'בקטסט', 'יומן מסחר', 'טעויות מנטליות', 'מה עוצר אותך', 'מחויבות לחודשיים',
  'מטרה ולמה עכשיו', 'ציון'
];

function doPost(e) {
  try {
    const d = JSON.parse(e.postData.contents);
    if (d.website) return json_({ ok: true }); // honeypot – בוטים

    const sheet = getSheet_();
    const score = score_(d);
    const row = [
      new Date(), d.name, "'" + d.phone, d.email, d.experience, d.status, d.failed,
      d.backtest, d.journal, [].concat(d.mistakes || []).join(', '), d.blocker, d.commitment,
      d.goal, score
    ];
    sheet.appendRow(row);

    MailApp.sendEmail({
      to: NOTIFY_EMAIL,
      subject: 'ליד חדש לליווי: ' + d.name + ' (ציון ' + score + '/6)',
      body: HEADERS.map(function (h, i) { return h + ': ' + String(row[i]).replace(/^'/, ''); }).join('\n')
    });
    return json_({ ok: true });
  } catch (err) {
    return json_({ ok: false, error: String(err) });
  }
}

function getSheet_() {
  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheets()[0];
  if (sheet.getLastRow() === 0) sheet.appendRow(HEADERS);
  sheet.setFrozenRows(1);
  sheet.setRightToLeft(true);
  return sheet;
}

// ציון סינון 0–6: בקטסט, יומן, מחויבות, ניסיון, תשובות פתוחות לא שטחיות
function score_(d) {
  let s = 0;
  if (d.backtest === 'כן, מסודר על 100+ עסקאות') s += 1;
  if (d.journal === 'תמיד') s += 1;
  if (d.commitment === 'כן, לגמרי') s += 2;
  if (d.experience === '2-4 שנים' || d.experience === '4+ שנים') s += 1;
  if ((d.blocker || '').length > 40 && (d.goal || '').length > 40) s += 1;
  return s;
}

function json_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}
