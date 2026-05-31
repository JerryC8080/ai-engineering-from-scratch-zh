import assert from 'node:assert';
// build.js 用运行时 `module.exports` 赋值,Node 的 CJS-interop 无法静态识别命名导出,
// 故用 default import 再解构。
import bjs from '../site/build.js';
const { extractLessonMetaForLang } = bjs;

// matrix-transformations 的 zh.md:首行是译注,真正 motto 在第二个 > 引用
const zh = extractLessonMetaForLang(
  'phases/01-math-foundations/03-matrix-transformations', 'zh');
assert.ok(zh.summary && !zh.summary.includes('译注'),
  'zh summary 应跳过译注行,取下一条引用; got: ' + zh.summary);
assert.ok(zh.keywords.includes('旋转'),
  'zh keywords 应含中文 H3; got: ' + zh.keywords);

const en = extractLessonMetaForLang(
  'phases/01-math-foundations/03-matrix-transformations', 'en');
assert.ok(en.summary.startsWith('A matrix'),
  'en summary 不变; got: ' + en.summary);

console.log('PASS: extractLessonMetaForLang zh/en');
