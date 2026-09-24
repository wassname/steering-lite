// Static results page: SVG Pareto plot, Pareto-best table, answer explorer. Reads ./points.json
// written by scripts/bsbench/results.py (the same data as plot.png and index.md). -- PI/Claude
import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const COLORS = {
  vjp_delta: '#0072b2', mean_diff: '#d55e00', pca: '#cc79a7', vjp_cache: '#009e73',
  kv_cache_gram: '#e69f00', prompting: '#6a3d9a', prompting_engineered: '#b15928', random: '#999999',
};
const W = 1000, H = 560, M = { l: 70, r: 20, t: 30, b: 50 };
const fmt = (x, d = 2) => (x == null ? '—' : (x >= 0 ? '+' : '') + x.toFixed(d));
const pointId = p => `${p.method}_s${p.seed}_${p.side}_C${p.C}`;
const directed = p => (p.side === '+C' ? p.effect : -p.effect);

function Plot({ data, selected, onSelect }) {
  const [hover, setHover] = useState(null);
  const shown = [...data.curves.flatMap(c => c.points), ...data.points.filter(p => p.admissible && (p.method === 'random' || p.method.startsWith('prompting')))];
  const xMax = 1.08 * Math.max(...shown.map(p => Math.abs(p.effect)), 0.5);
  const yMax = 1.08 * Math.max(...shown.map(p => p.off_axis), 0.3);
  const x = v => M.l + ((v + xMax) / (2 * xMax)) * (W - M.l - M.r);
  const y = v => M.t + ((v + 0.05) / (yMax + 0.05)) * (H - M.t - M.b);
  const zone = data.zone;
  const zonePath = zone.length > 1
    ? 'M' + [...zone.map(z => `${x(z[2])},${y(z[1])}`), ...[...zone].reverse().map(z => `${x(z[3])},${y(z[1])}`)].join('L') + 'Z' : null;
  const ticks = n => Array.from({ length: n + 1 }, (_, i) => i);
  const random = data.points.filter(p => p.method === 'random');
  return <div className="chart-shell">
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="judged on-axis change against off-axis damage">
      <rect className="canvas" width={W} height={H} />
      <g className="grid">
        {ticks(8).map(i => { const v = -xMax + (i * 2 * xMax) / 8; return <g key={`x${i}`}><line x1={x(v)} x2={x(v)} y1={M.t} y2={H - M.b} /><text x={x(v)} y={H - M.b + 16} textAnchor="middle">{v.toFixed(1)}</text></g>; })}
        {ticks(5).map(i => { const v = (i * yMax) / 5; return <g key={`y${i}`}><line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} /><text x={M.l - 6} y={y(v) + 4} textAnchor="end">{v.toFixed(1)}</text></g>; })}
      </g>
      <text className="axis" x={(W + M.l) / 2} y={H - 8} textAnchor="middle">judge on-axis change (left: abrasive / candid, right: sycophantic)</text>
      <text className="axis" transform={`translate(16 ${(H + M.t) / 2}) rotate(-90)`} textAnchor="middle">off-axis damage (lower is better)</text>
      {zonePath && <path d={zonePath} className="zone" />}
      {random.map(p => <circle key={pointId(p)} cx={x(p.effect)} cy={y(p.off_axis)} r={p.admissible ? 3 : 2} className={p.admissible ? 'random' : 'random rejected'} />)}
      {data.curves.filter(c => c.points.length).map(c => {
        const end = c.points.at(-1);
        return <g key={c.method + c.side}>
          <polyline points={c.path.map(([a, b]) => `${x(a)},${y(b)}`).join(' ')} fill="none" stroke={COLORS[c.method]} strokeWidth="2.5" strokeDasharray={c.side === '-C' ? '6 4' : ''} />
          {c.points.map(p => {
            const full = data.points.find(q => q.method === c.method && q.side === c.side && q.C === p.C);
            const isSel = selected && full && pointId(full) === pointId(selected);
            const isEnd = p === end;
            return isEnd
              ? <path key={p.C} d="M-6,-6L6,6M-6,6L6,-6" transform={`translate(${x(p.effect)} ${y(p.off_axis)})`} stroke={COLORS[c.method]} strokeWidth="3.5"
                  className="mark end" onPointerEnter={() => setHover({ ...p, method: c.method, side: c.side })} onPointerLeave={() => setHover(null)} onClick={() => full && onSelect(full)} />
              : <circle key={p.C} cx={x(p.effect)} cy={y(p.off_axis)} r={isSel ? 7 : 4} fill={COLORS[c.method]} fillOpacity={isSel ? 1 : 0.5} stroke={isSel ? '#000' : 'none'}
                  className="mark" onPointerEnter={() => setHover({ ...p, method: c.method, side: c.side })} onPointerLeave={() => setHover(null)} onClick={() => full && onSelect(full)} />;
          })}
          <text x={x(end.effect)} y={y(end.off_axis) - 11} textAnchor="middle" className="label" fill={COLORS[c.method]}>{c.method} {c.side}</text>
        </g>;
      })}
      {data.points.filter(p => p.method.startsWith('prompting')).map(p => <g key={pointId(p)} className="mark" onClick={() => onSelect(p)}
        onPointerEnter={() => setHover(p)} onPointerLeave={() => setHover(null)}>
        <path d="M0,-8 L2.4,-2.5 8,-2.5 3.5,1 5,7 0,3.5 -5,7 -3.5,1 -8,-2.5 -2.4,-2.5Z" transform={`translate(${x(p.effect)} ${y(p.off_axis)})`} fill={COLORS[p.method]} />
        <text x={x(p.effect)} y={y(p.off_axis) - 11} textAnchor="middle" className="label" fill={COLORS[p.method]}>{p.method === 'prompting' ? 'prompt' : 'eng. prompt'} {p.side}</text>
      </g>)}
      <path d="M0,-7 7,0 0,7 -7,0Z" transform={`translate(${x(0)} ${y(0)})`} fill="#333" /><text x={x(0) + 10} y={y(0) - 6} className="label">bare</text>
    </svg>
    {hover && <aside className="tooltip" style={{ left: `${(x(hover.effect) / W) * 100}%`, top: `${(y(hover.off_axis) / H) * 100}%` }}>
      <strong>{hover.method} {hover.side} C={hover.C.toPrecision(3)}</strong>
      <span>on-axis {fmt(hover.effect)}, off-axis {hover.off_axis.toFixed(2)}</span><span>click to open its answers</span>
    </aside>}
  </div>;
}

function Summary({ data }) {
  return <table>
    <thead><tr><th>method</th><th>score↑</th><th>90% CI</th><th>−C on↑</th><th>−C off↓</th><th>−C C</th><th>+C on↑</th><th>+C off↓</th><th>+C C</th><th>seeds</th><th>N</th><th>rejected</th></tr></thead>
    <tbody>{data.summary.map(r => <tr key={r.method}>
      <td className={r.method === 'random' || r.method.startsWith('prompting') ? 'control' : ''}><span className="swatch" style={{ background: COLORS[r.method] }} />{r.method}</td>
      <td><strong>{fmt(r.score)}</strong></td><td>{r.ci[0] == null ? '—' : `[${fmt(r.ci[0])}, ${fmt(r.ci[1])}]`}</td>
      {['-C', '+C'].flatMap(side => { const b = r.best[side]; return b ? [<td key={side + 'e'}>{fmt(side === '+C' ? b.effect : -b.effect)}</td>, <td key={side + 'o'}>{b.off_axis.toFixed(2)}</td>, <td key={side + 'c'}>{b.C.toPrecision(3)}</td>] : [<td key={side + 'e'}>—</td>, <td key={side + 'o'}>—</td>, <td key={side + 'c'}>—</td>]; })}
      <td>{r.seeds}</td><td>{r.N}</td><td>{r.rejected}</td>
    </tr>)}</tbody>
  </table>;
}

function Explorer({ data, selected, onSelect }) {
  const [scenario, setScenario] = useState(data.questions[0].scenario);
  const question = data.questions.find(q => q.scenario === scenario);
  // default rows: each method-side at its Pareto-best dose; the selected point is added on top
  const rows = useMemo(() => {
    const best = data.summary.flatMap(r => ['-C', '+C'].map(side => r.best[side] && data.points.find(p => p.method === r.method && p.side === side && p.C === r.best[side].C && (r.method !== 'random' || p.seed === 0))))
      .filter(Boolean);
    return selected && !best.some(p => pointId(p) === pointId(selected)) ? [selected, ...best] : best;
  }, [data, selected]);
  return <section>
    <label className="picker">question <select value={scenario} onChange={e => setScenario(e.target.value)}>
      {data.questions.map(q => <option key={q.scenario} value={q.scenario}>{q.scenario.replace('syco_bullshit_v2_', '')}: {q.prompt.slice(0, 90)}…</option>)}
    </select></label>
    <div className="question"><p><strong>Question.</strong> {question.prompt}</p><p className="flaw"><strong>Known flaw (given to the aware judge, not the blind one).</strong> {question.flaw}</p></div>
    <div className="answer bare"><h3>bare</h3><p>{question.bare}</p></div>
    {rows.map(p => {
      const q = p.questions.find(q => q.scenario === scenario);
      return <div key={pointId(p)} className={`answer ${selected && pointId(p) === pointId(selected) ? 'selected' : ''}`} style={{ borderLeftColor: COLORS[p.method] }}>
        <h3>{p.method} s{p.seed} {p.side} C={p.C.toPrecision(3)} {p.admissible ? '' : <em>(not admissible)</em>}</h3>
        {q ? <>
          <p>{q.text}</p>
          <p className="judge">aware judge: on-axis {fmt(q.effect)} (toward {p.side === '+C' ? 'sycophancy' : 'candour'}: {fmt(p.side === '+C' ? q.effect : -q.effect)}), off-axis {q.off_axis.toFixed(2)}. “{q.evidence}”</p>
          {q.blind && <p className="judge">blind judge: <strong>{q.blind.concept}</strong> ({q.blind.strength}) — {q.blind.change}</p>}
        </> : <p>no judged answer for this question</p>}
      </div>;
    })}
  </section>;
}

function App() {
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState(null);
  useEffect(() => { fetch('points.json').then(r => r.json()).then(setData); }, []);
  if (!data) return <main><p>loading points.json…</p></main>;
  return <main>
    <h1>steering-lite on Bullshit Bench v2</h1>
    <p className="lede">How far can each steering method push a model toward or away from sycophancy before the answers break? Explanation below the plot.</p>
    <Plot data={data} selected={selected} onSelect={p => { setSelected(p); document.getElementById('explorer').scrollIntoView({ behavior: 'smooth' }); }} />
    {/* intro text: PI/Claude, rewrite freely */}
    <section className="intro">
      <p>We add a steering vector to a small language model ({data.model_dir.split('-g')[0].replace('--', '/')}) and ask it {data.questions.length} questions from Bullshit Bench v2.
        Each question rests on a made-up premise, such as the thermal conductivity of a CI pipeline. A good answer points out the made-up part.
        Steering one way (+C) should make the model go along with the premise (sycophantic). Steering the other way (−C) should make it point out the problem (candid).</p>
      <p>Each colour is one method. We raise the steering strength step by step until the answers stop making sense.
        Left to right is how far a judge model says the answers moved: right is more sycophantic, left is more candid.
        Up and down is other damage the judge saw, such as rambling or going off topic; higher on the page is better.
        The line joins each method's best trade-offs and ends at its last sensible strength (×). Faint dots are the other strengths we tried.
        Solid lines are +C, dashed lines are −C. Stars are plain prompts, for example "Answer as someone who is sycophantic".
        Grey dots are random directions with the same effect on the model's outputs; the grey band is where most of them land.</p>
    </section>
    <h2>Best strength per method</h2>
    <p className="lede">For each side we pick the strength with the best on-axis gain minus {data.off_weight}× damage, then score the method by its weaker side. The 90% range comes from resampling the questions.</p>
    <Summary data={data} />
    <h2 id="explorer">Answers</h2>
    <p className="lede">Pick a question. Each block is one method and side at its Pareto-best dose (random: seed 0). Click a point on the plot to add that dose.</p>
    <Explorer data={data} selected={selected} onSelect={setSelected} />
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
