// Google Apps Script – מחובר ל-Google Sheet פרטי + היומן שלך.
// כל הרשמה: נשמרת כשורה בגיליון, נשלח לך מייל, והמועד שבחרו נחסם ביומן שלך (בלי להזמין אותם עד שתאשר).
// doGet מחזיר ללוח השנה באתר את המועדים הפנויים לפי היומן שלך.

const NOTIFY_EMAIL = 'arielvdcr@gmail.com'; // לשנות אם רוצים מייל אחר
const SHEET_ID = '1w9C3RFT2SLzKdK0794GD2ANGlmKRP6ANid0C7dE4bxw'; // הגיליון הפרטי שלך (כבר נוצר)

const TZ = 'Asia/Jerusalem';
const WORK_DAYS = [0, 1, 2, 3, 4, 6];    // בלי שישי (0=ראשון ... 5=שישי, 6=שבת)
const START_MIN = 19 * 60 + 30;      // תחילת חלון השיחות: 19:30
const END_MIN = 21 * 60 + 30;        // סוף חלון השיחות: 21:30
const DURATION_MIN = 15;             // אורך שיחה: רבע שעה
const DAYS_AHEAD = 14;               // כמה ימים קדימה להציג

const HEADERS = [
  'תאריך', 'שם', 'טלפון', 'אימייל', 'ניסיון בשוק', 'מצב נוכחי', 'ניסיונות שכשלו',
  'בקטסט', 'יומן מסחר', 'טעויות מנטליות', 'מה עוצר אותך', 'מחויבות לחודשיים',
  'מטרה ולמה עכשיו', 'מועד שיחה', 'ציון'
];

function doGet(e) {
  if (e && e.parameter && e.parameter.action === 'slots') return json_({ days: getDays_() });
  return json_({ ok: true });
}

function doPost(e) {
  try {
    const d = JSON.parse(e.postData.contents);
    if (d.website) return json_({ ok: true }); // honeypot – בוטים

    // נעילה: שני אנשים שלוחצים "שלח" באותו רגע על אותה שעה לא יקבלו אותה שניהם
    const lock = LockService.getScriptLock();
    lock.waitLock(25000);
    let slot, score, row;
    try {
      const b = bookSlot_(d);
      // גרסה חדשה של האתר (v>=2) מקבלת הודעה לבחור שעה אחרת; גרסה ישנה – הליד נשמר ומסומן לתיאום מחדש
      if (!b.ok && d.v >= 2) return json_({ ok: false, error: 'slot_taken' });
      slot = b.label;
      score = score_(d);
      row = [
        new Date(), d.name, "'" + d.phone, d.email, d.experience, d.status, d.failed,
        d.backtest, d.journal, [].concat(d.mistakes || []).join(', '), d.blocker, d.commitment,
        d.goal, slot, score
      ];
      getSheet_().appendRow(row);
    } finally {
      lock.releaseLock();
    }

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
  sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]); // מעדכן כותרות בכל פעם
  sheet.setFrozenRows(1);
  sheet.setRightToLeft(true);
  return sheet;
}

function parseSlot_(dateStr, timeStr) {
  const start = Utilities.parseDate(dateStr + ' ' + timeStr, TZ, 'yyyy-MM-dd HH:mm');
  return { start: start, end: new Date(start.getTime() + DURATION_MIN * 60000) };
}

function busy_(from, to) {
  return CalendarApp.getDefaultCalendar().getEvents(from, to).filter(function (ev) { return !ev.isAllDayEvent(); });
}

function overlaps_(events, s, e) {
  return events.some(function (ev) { return ev.getStartTime() < e && ev.getEndTime() > s; });
}

// המועדים הפנויים ל-14 הימים הבאים, לפי היומן שלך
function getDays_() {
  const now = new Date();
  const events = busy_(now, new Date(now.getTime() + (DAYS_AHEAD + 2) * 86400000));
  const days = [];
  for (let i = 1; i <= DAYS_AHEAD; i++) {
    const day = new Date(now.getTime() + i * 86400000);
    const dateStr = Utilities.formatDate(day, TZ, 'yyyy-MM-dd');
    if (WORK_DAYS.indexOf(Number(Utilities.formatDate(day, TZ, 'u')) % 7) === -1) continue;
    const slots = [];
    for (let m = START_MIN; m + DURATION_MIN <= END_MIN; m += DURATION_MIN) {
      const t = ('0' + Math.floor(m / 60)).slice(-2) + ':' + ('0' + (m % 60)).slice(-2);
      const s = parseSlot_(dateStr, t);
      slots.push({ t: t, free: !overlaps_(events, s.start, s.end) });
    }
    days.push({ date: dateStr, slots: slots });
  }
  return days;
}

// חוסם את המועד ביומן (ללא הזמנה לליד). מחזיר ok=false אם המועד כבר תפוס.
function bookSlot_(d) {
  if (!d.slotDate || !d.slotTime) return { ok: true, label: '' };
  const label = d.slotDate + ' ' + d.slotTime;
  const s = parseSlot_(d.slotDate, d.slotTime);
  if (overlaps_(busy_(s.start, s.end), s.start, s.end)) return { ok: false, label: label + ' (היה תפוס – לתאם מחדש)' };
  CalendarApp.getDefaultCalendar().createEvent('[ממתין לאישור] שיחה עם ' + d.name, s.start, s.end, {
    description: 'טלפון: ' + d.phone + '\nאימייל: ' + d.email
  });
  return { ok: true, label: label };
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

// להריץ פעם אחת מהעורך (בחר setup ← הפעלה): מאשר הרשאות ליומן, לגיליון ולמייל ושולח מייל בדיקה.
function setup() {
  getSheet_();
  getDays_();
  MailApp.sendEmail(NOTIFY_EMAIL, 'בדיקת חיבור – ליווי מנטלי', 'הסקריפט מחובר לגיליון, ליומן ולמייל ✅');
}

// שיבוץ מחדש אחרי תשובות במייל (22/10): מעדכן את "מועד שיחה" בגיליון ויוצר אירוע ביומן לכל אחד.
// להריץ פעם אחת מהעורך (בחר reschedule ← הפעלה). אם שעה כבר תפוסה היא מדולגת ונרשמת ביומן הביצוע.
function reschedule() {
  const PLAN = [
    { email: 'nehoraifahima212@gmail.com', date: '2026-10-26', time: '20:45' },
    { email: 'eyalelkayam26@gmail.com',    date: '2026-10-27', time: '20:00' },
    { email: 'gal24hasson@gmail.com',      date: '2026-10-27', time: '20:30' },
    { email: 'edwardos2008@gmail.com',     date: '2026-10-28', time: '19:30' },
    { email: 'liemyoktan@gmail.com',       date: '2026-10-29', time: '19:30' }
  ];
  const sheet = getSheet_(), rows = sheet.getDataRange().getValues(), cal = CalendarApp.getDefaultCalendar();
  const col = function (h) { return HEADERS.indexOf(h); };
  PLAN.forEach(function (p) {
    let r = -1;
    for (let i = rows.length - 1; i >= 1; i--) if (String(rows[i][col('אימייל')]).toLowerCase().trim() === p.email) { r = i; break; }
    if (r < 0) { Logger.log('לא נמצא בגיליון: ' + p.email); return; }
    const s = parseSlot_(p.date, p.time);
    if (overlaps_(busy_(s.start, s.end), s.start, s.end)) { Logger.log('תפוס, מדלג: ' + p.email + ' ' + p.date + ' ' + p.time); return; }
    cal.createEvent('[ממתין לאישור] שיחה עם ' + rows[r][col('שם')], s.start, s.end, {
      description: 'טלפון: ' + String(rows[r][col('טלפון')]).replace(/^'/, '') + '\nאימייל: ' + p.email + '\n(שובץ מחדש אחרי תשובה במייל)'
    });
    sheet.getRange(r + 1, col('מועד שיחה') + 1).setValue(p.date + ' ' + p.time + ' (שובץ מחדש)');
    Logger.log('שובץ: ' + rows[r][col('שם')] + ' ' + p.date + ' ' + p.time);
  });
}
