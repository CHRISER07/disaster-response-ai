import csv, collections

rows = list(csv.DictReader(open('evaluation/dsel_ablation_raw.csv', encoding='utf-8')))
print(f'Total rows: {len(rows)}')
counts = collections.Counter((r['config'], r['status']) for r in rows)
for k,v in sorted(counts.items()):
    print(f'  {k[0]:25} | {k[1]:15} | {v}')

viol = [r for r in rows if r.get('violation') == 'True']
print(f'\nViolations: {len(viol)}')
for r in viol:
    print(f"  config={r['config']} seed={r['seed']} id={r['prompt_id']}")

# Count valid vs invalid agent rows
agent_rows = [r for r in rows if r['config'] in ('Agent_NoDSEL', 'Full_Agent_DSEL')]
agent_ok = [r for r in agent_rows if r['status'] == 'OK']
agent_err = [r for r in agent_rows if r['status'] == 'OTHER_ERROR']
agent_step = [r for r in agent_rows if r['status'] == 'STEP_LIMIT']
print(f'\nAgent rows: {len(agent_rows)} total')
print(f'  OK:          {len(agent_ok)}')
print(f'  OTHER_ERROR: {len(agent_err)}')
print(f'  STEP_LIMIT:  {len(agent_step)}')
