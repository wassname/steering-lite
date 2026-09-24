// Static results page: SVG Pareto plot, Pareto-best table, answer explorer. Reads ./points.json
// written by scripts/bsbench/results.py (the same data as plot.png and index.md). -- PI/Claude
import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const COLORS = {
  vjp_delta: '#0072b2', mean_diff: '#d55e00', pca: '#cc79a7', vjp_cache: '#009e73',
  kv_cache_gram: '#e69f00', prompting: '#6a3d9a', random: '#999999',
};
const W = 1000, H = 560, M = { l: 70, r: 20, t: 30, b: 50 };
const fmt = (x, d = 2) => (x == null ? '—' : (x >= 0 ? '+' : '') + x.toFixed(d));
const pointId = p => `${p.method}_s${p.seed}_${p.side}_C${p.C}`;
const directed = p => (p.side === '+C' ? p.effect : -p.effect);

function Plot({ data, selected, onSelect }) {
  const [hover, setHover] = useState(null);
  const shown = [...data.curves.flatMap(c => c.points), ...data.points.filter(p => p.admissible && (p.method === 'random' || p.method === 'prompting'))];
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
        const pts = [[0, 0], ...c.points.map(p => [p.effect, p.off_axis])];
        return <g key={c.method + c.side}>
          <polyline points={pts.map(([a, b]) => `${x(a)},${y(b)}`).join(' ')} fill="none" stroke={COLORS[c.method]} strokeWidth="2.5" strokeDasharray={c.side === '-C' ? '6 4' : ''} />
          {c.points.map(p => {
            const full = data.points.find(q => q.method === c.method && q.side === c.side && q.C === p.C);
            const isSel = selected && full && pointId(full) === pointId(selected);
            return <circle key={p.C} cx={x(p.effect)} cy={y(p.off_axis)} r={isSel ? 7 : 4.5} fill={COLORS[c.method]} stroke={isSel ? '#000' : 'white'}
              className="mark" onPointerEnter={() => setHover({ ...p, method: c.method, side: c.side })} onPointerLeave={() => setHover(null)} onClick={() => full && onSelect(full)} />;
          })}
          <text x={x(c.points.at(-1).effect)} y={y(c.points.at(-1).off_axis) - 9} textAnchor="middle" className="label" fill={COLORS[c.method]}>{c.method} {c.side}</text>
        </g>;
      })}
      {data.points.filter(p => p.method === 'prompting').map(p => <g key={pointId(p)} className="mark" onClick={() => onSelect(p)}
        onPointerEnter={() => setHover(p)} onPointerLeave={() => setHover(null)}>
        <path d="M0,-8 L2.4,-2.5 8,-2.5 3.5,1 5,7 0,3.5 -5,7 -3.5,1 -8,-2.5 -2.4,-2.5Z" transform={`translate(${x(p.effect)} ${y(p.off_axis)})`} fill={COLORS.prompting} />
        <text x={x(p.effect)} y={y(p.off_axis) - 11} textAnchor="middle" className="label" fill={COLORS.prompting}>prompt {p.side}</text>
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
      <td className={r.method === 'random' || r.method === 'prompting' ? 'control' : ''}><span className="swatch" style={{ background: COLORS[r.method] }} />{r.method}</td>
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
    <p className="lede">{data.model_dir}, cohort <strong>{data.cohort}</strong> ({data.questions.length} questions), judge {data.judge}.
      Each line is one method's dose walk from bare (◆) to its last coherent dose; solid = +C (toward sycophancy), dashed = −C (toward candour).
      Grey = random directions. Score = min over ±C of on-axis − {data.off_weight}×off-axis at each side's best admissible dose; CI from a bootstrap over questions.</p>
    <Plot data={data} selected={selected} onSelect={p => { setSelected(p); document.getElementById('explorer').scrollIntoView({ behavior: 'smooth' }); }} />
    <h2>Pareto-best dose per method</h2>
    <Summary data={data} />
    <h2 id="explorer">Answers</h2>
    <p className="lede">Pick a question. Each block is one method and side at its Pareto-best dose (random: seed 0). Click a point on the plot to add that dose.</p>
    <Explorer data={data} selected={selected} onSelect={setSelected} />
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
