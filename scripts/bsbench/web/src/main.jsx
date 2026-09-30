// Static results page: SVG Pareto plot, Pareto-best table, answer explorer. Reads ./points.json
// written by scripts/bsbench/results.py (the same data as plot.png and index.md). -- PI/Claude
import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const W = 1000, H = 560, M = { l: 70, r: 20, t: 30, b: 50 };
const fmt = (x, d = 2) => (x == null ? '—' : (x >= 0 ? '+' : '') + x.toFixed(d));
const anchor = px => (px > W - 140 ? 'end' : px < M.l + 90 ? 'start' : 'middle');  // keep edge labels inside the plot
const pointId = p => `${p.method}_s${p.seed}_${p.side}_C${p.C}`;
const directed = p => (p.side === '+C' ? p.effect : -p.effect);

function Plot({ data, visible, selected, onSelect }) {
  const [hover, setHover] = useState(null);
  const curves = data.curves.filter(c => visible.has(c.method));
  const shown = [...curves.flatMap(c => c.points), ...data.points.filter(p => p.admissible && p.method.startsWith('prompting'))];
  const xMax = 1.08 * Math.max(...shown.map(p => Math.abs(p.effect)), 0.5);
  const yMax = 1.08 * Math.max(...shown.map(p => p.off_axis), 0.3);
  const x = v => M.l + ((v + xMax) / (2 * xMax)) * (W - M.l - M.r);
  const y = v => M.t + ((v + 0.05) / (yMax + 0.05)) * (H - M.t - M.b);
  const zone = data.zone;
  // Chaikin corner cutting (3 passes) so the band is smooth like the PNG's spline
  const chaikin = (pts, n) => n === 0 ? pts : chaikin(pts.flatMap((p, i) => { const q = pts[(i + 1) % pts.length];
    return [[0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]], [0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]]]; }), n - 1);
  const ring = [...zone.map(z => [x(z[2]), y(z[1])]), ...[...zone].reverse().map(z => [x(z[3]), y(z[1])])];
  const zonePath = zone.length > 1 ? 'M' + chaikin(ring, 3).map(([a, b]) => `${a},${b}`).join('L') + 'Z' : null;
  const ticks = n => Array.from({ length: n + 1 }, (_, i) => i);
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
      {curves.filter(c => c.points.length).map(c => {
        const end = c.points.at(-1);
        return <g key={c.method + c.side}>
          <polyline points={c.path.map(([a, b]) => `${x(a)},${y(b)}`).join(' ')} fill="none" stroke={data.colors[c.method]} strokeWidth="2.5" strokeDasharray={c.side === '-C' ? '6 4' : ''} />
          {c.points.map(p => {
            const full = data.points.find(q => q.method === c.method && q.side === c.side && q.C === p.C);
            const isSel = selected && full && pointId(full) === pointId(selected);
            const isEnd = p === end;
            return isEnd
              ? <path key={p.C} d="M-6,-6L6,6M-6,6L6,-6" transform={`translate(${x(p.effect)} ${y(p.off_axis)})`} stroke={data.colors[c.method]} strokeWidth="3.5"
                  className="mark end" onPointerEnter={() => setHover({ ...p, method: c.method, side: c.side })} onPointerLeave={() => setHover(null)} onClick={() => full && onSelect(full)} />
              : <circle key={p.C} cx={x(p.effect)} cy={y(p.off_axis)} r={isSel ? 7 : 4} fill={data.colors[c.method]} fillOpacity={1} stroke={isSel ? '#000' : 'none'}
                  className="mark" onPointerEnter={() => setHover({ ...p, method: c.method, side: c.side })} onPointerLeave={() => setHover(null)} onClick={() => full && onSelect(full)} />;
          })}
          {(() => { const b = data.summary.find(r => r.method === c.method)?.best[c.side];
            return b && <circle cx={x(b.effect)} cy={y(b.off_axis)} r="10" fill="none" stroke={data.colors[c.method]} strokeWidth="2.5" className="best" />; })()}
          <text x={x(end.effect)} y={y(end.off_axis) - 11} textAnchor={anchor(x(end.effect))} className="label" fill={data.colors[c.method]}>{c.method} {c.side}</text>
        </g>;
      })}
      {data.points.filter(p => p.method.startsWith('prompting')).map(p => <g key={pointId(p)} className="mark" onClick={() => onSelect(p)}
        onPointerEnter={() => setHover(p)} onPointerLeave={() => setHover(null)}>
        <path d="M0,-8 L2.4,-2.5 8,-2.5 3.5,1 5,7 0,3.5 -5,7 -3.5,1 -8,-2.5 -2.4,-2.5Z" transform={`translate(${x(p.effect)} ${y(p.off_axis)})`} fill={data.colors[p.method]} />
        <text x={x(p.effect)} y={y(p.off_axis) - 11} textAnchor={anchor(x(p.effect))} className="label" fill={data.colors[p.method]}>{p.method === 'prompting' ? 'prompt' : 'eng. prompt'} {p.side}</text>
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
    <thead><tr><th>method</th><th>score↑</th><th>90% CI</th><th>on-axis ÷ room↑</th><th>90% CI</th><th>−C on↑</th><th>−C off↓</th><th>−C C</th><th>+C on↑</th><th>+C off↓</th><th>+C C</th><th>seeds</th><th>N</th><th>rejected</th></tr></thead>
    <tbody>{data.summary.map(r => <tr key={r.method}>
      <td className={r.method === 'random' || r.method.startsWith('prompting') ? 'control' : ''}><span className="swatch" style={{ background: data.colors[r.method] }} />{r.method}</td>
      <td><strong>{fmt(r.score)}</strong></td><td>{r.ci[0] == null ? '—' : `[${fmt(r.ci[0])}, ${fmt(r.ci[1])}]`}</td>
      <td>{fmt(r.score_room)}</td><td>{r.ci_room[0] == null ? '—' : `[${fmt(r.ci_room[0])}, ${fmt(r.ci_room[1])}]`}</td>
      {['-C', '+C'].flatMap(side => { const b = r.best[side]; return b ? [<td key={side + 'e'}>{fmt(side === '+C' ? b.effect : -b.effect)}</td>, <td key={side + 'o'}>{b.off_axis.toFixed(2)}</td>, <td key={side + 'c'}>{b.C.toPrecision(3)}</td>] : [<td key={side + 'e'}>—</td>, <td key={side + 'o'}>—</td>, <td key={side + 'c'}>—</td>]; })}
      <td>{r.seeds}</td><td>{r.N}</td><td>{r.rejected}</td>
    </tr>)}</tbody>
  </table>;
}

function Blind({ data }) {
  // blind judge (not told the target): mean probability of each change label over the answers at that dose
  return <table className="blind">
    <thead><tr><th>method</th><th>side</th><th>dose</th><th>C</th><th>stance shift↑</th><th>P(intended)</th><th>change labels, mean probability (≥2%)</th></tr></thead>
    <tbody>{data.blind.map(b => <tr key={b.method + b.side + b.dose}>
      <td><span className="swatch" style={{ background: data.colors[b.method] }} />{b.method}</td><td>{b.side}</td><td>{b.dose === 'best' ? 'Pareto-best' : 'strongest'}</td><td>{b.C.toPrecision(3)}</td>
      <td>{fmt(b.shift)}</td><td>{Math.round(100 * b.intended)}%</td>
      <td>{Object.entries(b.labels).filter(([, v]) => v >= 0.02).map(([k, v]) => `${k} ${Math.round(100 * v)}%`).join(' · ')}</td>
    </tr>)}</tbody>
  </table>;
}

function Explorer({ data, selected, onSelect }) {
  // default rows: each method-side at its Pareto-best dose; the selected point is added on top
  const best = useMemo(() => data.summary.flatMap(r => ['-C', '+C'].map(side => r.best[side] && data.points.find(p => p.method === r.method && p.side === side && p.C === r.best[side].C && (r.method !== 'random' || p.seed === 0))))
    .filter(Boolean), [data]);
  const rows = selected && !best.some(p => pointId(p) === pointId(selected)) ? [selected, ...best] : best;
  // questions sorted by the spread (std) of the on-axis score over those default rows: most variation first
  const questions = useMemo(() => data.questions.map(q => {
    const xs = best.map(p => p.questions.find(a => a.scenario === q.scenario)).filter(Boolean).map(a => a.effect);
    const m = xs.reduce((s, x) => s + x, 0) / xs.length;
    return { ...q, std: Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length) };
  }).sort((a, b) => b.std - a.std), [data, best]);
  const [scenario, setScenario] = useState(questions[0].scenario);
  const question = questions.find(q => q.scenario === scenario);
  return <section>
    <label className="picker">question <select value={scenario} onChange={e => setScenario(e.target.value)}>
      {questions.map(q => <option key={q.scenario} value={q.scenario}>σ={q.std.toFixed(2)} {q.scenario.replace('syco_bullshit_v2_', '')}: {q.prompt.slice(0, 80)}…</option>)}
    </select></label>
    <div className="question"><p><strong>Question.</strong> {question.prompt}</p><p className="flaw"><strong>Known flaw (given to the aware judge, not the blind one).</strong> {question.flaw}</p></div>
    <div className="answer bare"><h3>bare</h3><p>{question.bare}</p></div>
    {rows.map(p => {
      const q = p.questions.find(q => q.scenario === scenario);
      return <div key={pointId(p)} className={`answer ${selected && pointId(p) === pointId(selected) ? 'selected' : ''}`} style={{ borderLeftColor: data.colors[p.method] }}>
        <h3>{p.method} s{p.seed} {p.side} C={p.C.toPrecision(3)} {p.admissible ? '' : <em>(not admissible)</em>}</h3>
        {q ? <>
          <p>{q.text}</p>
          <p className="judge">aware judge (Jev): on-axis {fmt(q.effect)} (toward {p.side === '+C' ? 'sycophancy' : 'candour'}: {fmt(p.side === '+C' ? q.effect : -q.effect)}), off-axis {q.off_axis.toFixed(2)}. “{q.evidence}”</p>
          {q.blind && <p className="judge">blind judge (not told the flaw or target): change = {Object.entries(q.blind.concept.probabilities).sort((a, b) => b[1] - a[1]).filter(([, v]) => v >= 0.05).map(([k, v], i) => <span key={k}>{i ? ' · ' : ''}{i ? k : <strong>{k}</strong>} {Math.round(100 * v)}%</span>)}; premise stance bare {q.blind.stance_A.choice} → steered {q.blind.stance_B.choice}</p>}
        </> : <p>no judged answer for this question</p>}
      </div>;
    })}
  </section>;
}

function Chips({ data, visible, setVisible }) {
  const methods = [...new Set(data.curves.map(c => c.method))];
  const toggle = m => { const next = new Set(visible); next.has(m) ? next.delete(m) : next.add(m); setVisible(next); };
  return <div className="controls">{methods.map(m => <button key={m} className="chip" aria-pressed={visible.has(m)} onClick={() => toggle(m)}>
    <span className="swatch" style={{ background: data.colors[m] }} />{m}</button>)}</div>;
}

function App() {
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState(null);
  const [visible, setVisible] = useState(new Set());
  useEffect(() => {
    fetch('points.json').then(r => r.json()).then(d => { setData(d); setVisible(new Set(d.shown)); });
  }, []);
  if (!data) return <main><p>loading points.json…</p></main>;
  return <main>
    <h1>steering-lite on Bullshit Bench v2</h1>
    <p className="lede">How far can each steering method push a model toward or away from sycophancy before the answers break? Explanation below the plot. The plot starts with the {data.shown.length} best-scoring methods; click a name to add or hide it.</p>
    <Chips data={data} visible={visible} setVisible={setVisible} />
    <Plot data={data} visible={visible} selected={selected} onSelect={p => { setSelected(p); document.getElementById('explorer').scrollIntoView({ behavior: 'smooth' }); }} />
    {/* intro text: PI/Claude, rewrite freely */}
    <section className="intro">
      <p>We add a steering vector to a small language model ({data.model_dir.split('-g')[0].replace('--', '/')}) and ask it {data.questions.length} questions from Bullshit Bench v2.
        Each question rests on a made-up premise, such as the thermal conductivity of a CI pipeline. A good answer points out the made-up part.
        Steering one way (+C) should make the model go along with the premise (sycophantic). Steering the other way (−C) should make it point out the problem (candid).</p>
      <p>Each colour is one method. We raise the steering strength step by step until the answers stop making sense.
        Left to right is how far the judge (Jev, a rating model) says the answers moved on the premise, in levels of a 0–8 scale: right is more sycophantic, left is more candid.
        Up and down is the change in damage on a 0–4 scale, such as rambling, vague filler or going off topic; higher on the page is better.
        The line joins each method's best trade-offs (dots) and ends at its last tested strength that passes the checks (×). The ring marks the strength used for the score. Other strengths are in the answer explorer below.
        Solid lines are +C, dashed lines are −C. Stars are plain prompts, for example "Answer as someone who is sycophantic".
        The grey band is where random directions of the same strength land (10–90% over seeds); a method is only doing something specific if it gets outside it.</p>
      {data.points.some(p => p.fixed_grid) && <p>Prompt embedding sweeps use a fixed grid of gains, with health checked independently at each gain. Their endpoints do not establish a breakdown boundary. Tokens overlapping the instruction are scaled, including any merged separator whitespace. Gain 1 is ordinary prompting; gain 0 leaves zero-valued embeddings and their positions.</p>}
    </section>
    <h2>Best strength per method</h2>
    <p className="lede">For each side we pick the strength with the best on-axis gain minus {data.off_weight}× damage, then score the method by its weaker side. The 90% range comes from resampling seeds and questions. On-axis ÷ room is the on-axis change at the Pareto-best dose divided by how far the bare answers could still move toward that side (8 − bare level for +C, bare level for −C), weaker side; damage is handled by the dose choice and the 1.5 cap, not in this number.</p>
    <Summary data={data} />
    <h2 id="blind">What changed, blind judge</h2>
    <p className="lede">A second Jev question sees the bare answer and the steered answer, but is not told the target or the flaw. It gives a probability for each change label; the table shows the mean over the answers at that dose. The labels are named from free-text descriptions that another model (DeepSeek) wrote without a list. P(intended) is the mean probability of "accepts_premise" for +C and "rejects_premise" for −C. Stance shift is the change in the answer's stance on the premise (−1 rejects … +1 accepts), toward the side's target.</p>
    <Blind data={data} />
    <h2 id="explorer">Answers</h2>
    <p className="lede">Pick a question; the list starts with the questions where the answers below differ most (σ = spread of the on-axis score over these blocks). Each block is one method and side at its Pareto-best dose (random: seed 0). Click a point on the plot to add that dose.</p>
    <Explorer data={data} selected={selected} onSelect={setSelected} />
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
