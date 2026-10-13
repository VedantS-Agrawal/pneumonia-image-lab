import assert from 'node:assert/strict';
import fs from 'node:fs';
import {forward,conv,relu,pool} from '../dist/inference.js';
// The fixture is raw pixels exported by Python; avoids an image-decoding discrepancy.
const model=JSON.parse(fs.readFileSync('dist/assets/model-28.json'));
const fixture=JSON.parse(fs.readFileSync('tests/parity-fixtures.json'));
let max=0;
for(const test of fixture){const out=forward(test.pixels,model);const error=Math.abs(out.score-test.score);max=Math.max(max,error);assert.ok(error<1e-5,`Test ${test.id}: score error ${error}`);assert.equal(out.layers.at(-1).c,24);}
assert.deepEqual(Array.from(relu({data:new Float32Array([-2,0,3]),h:1,w:3,c:1}).data),[0,0,3]);
assert.deepEqual(Array.from(pool({data:new Float32Array([1,4,3,2]),h:2,w:2,c:1}).data),[4]);
const m=JSON.parse(fs.readFileSync('dist/assets/results.json'));
assert.equal(m.test_labels.length,624);assert.equal(m.test_scores['28'].length,624);
assert.deepEqual(m.config.sizes,[28]);
console.log(`Passed ${fixture.length} Python/browser parity cases. Max score error ${max.toExponential(3)}. Layer math and dataset contracts passed.`);
