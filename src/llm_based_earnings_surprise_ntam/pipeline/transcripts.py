"""CIQ earnings-call transcripts: pick one version per call, pull its text, clean.

Version rule (config transcripts.version_rule):
  pit   - the first version created for the call (usually a same-day Preliminary copy). This is the text
          you would actually have had in real time, so it is the default for backtests.
  final - the Final version, most recently created (cleanest text, but can appear days or weeks later).
Either way, `created_utc` records when the chosen version became available.
"""
import numpy as np
import pandas as pd

from .config import connect

EARNINGS_CALL = 48  # ciq keydeveventtypeid
ORDER = {
    'pit': 'v.created_utc asc nulls last, v.transcriptid asc',
    'final': '(v.transcriptpresentationtypeid = 5) desc, v.created_utc desc nulls last, v.transcriptid desc',
}
SPEAKER = {1: 'Operator', 2: 'Executive', 3: 'Analyst', 4: 'Shareholder', 5: 'Attendee', 6: 'Unknown'}
PRESENTATION = {1, 2, 5}       # operator message, presenter speech, presentation section
QA = {3, 4, 6, 7, 8}           # question, answer, Q&A section / operator / unknown Q&A message

META_SQL = """
    with v as (
        select d.keydevid::bigint as keydevid, d.transcriptid::bigint as transcriptid,
               d.companyid::bigint as companyid, d.companyname, d.headline,
               d.mostimportantdateutc as call_date, d.mostimportanttimeutc::text as call_time_utc,
               d.transcriptpresentationtypeid, d.transcriptpresentationtypename as presentation_type,
               d.transcriptcollectiontypename as collection_type,
               d.transcriptcreationdate_utc + coalesce(d.transcriptcreationtime_utc, '00:00'::time) as created_utc,
               d.audiolengthsec::float as audio_sec, g.isocountry2 as country
        from ciq_transcripts.wrds_transcript_detail d
        left join ciq.ciqcompany c on c.companyid = d.companyid
        left join ciq.ciqcountrygeo g on g.countryid = c.countryid
        where d.keydeveventtypeid = %(event)s
          and d.mostimportantdateutc >= %(start)s and d.mostimportantdateutc < %(end)s {country}
    ), n as (
        select keydevid, count(*) as n_versions, min(created_utc) as first_created_utc from v group by 1
    )
    select distinct on (v.keydevid) v.*, n.n_versions, n.first_created_utc
    from v join n using (keydevid)
    order by v.keydevid, {order}
"""

TEXT_SQL = """
    select c.transcriptid::bigint as transcriptid, c.componentorder, c.transcriptcomponenttypeid as component_type,
           p.transcriptpersonname as speaker, p.speakertypeid, c.componenttext
    from ciq_transcripts.ciqtranscriptcomponent c
    left join ciq_transcripts.ciqtranscriptperson p on p.transcriptpersonid = c.transcriptpersonid
    where c.transcriptid = any(%(ids)s)
"""


def assemble(comp):
    """Speaker turns -> one row per transcript with full, presentation and Q&A text plus simple counts."""
    comp = comp.sort_values(['transcriptid', 'componentorder'])
    role = comp.speakertypeid.map(SPEAKER).fillna('Unknown')
    name = comp.speaker.fillna(role)
    comp = comp.assign(line=name + ' (' + role + '): ' + comp.componenttext.fillna('').str.strip(),
                       words=comp.componenttext.fillna('').str.split().str.len(),
                       section=np.where(comp.component_type.isin(PRESENTATION), 'presentation',
                                        np.where(comp.component_type.isin(QA), 'qa', 'other')))
    g = comp.groupby('transcriptid')
    out = pd.DataFrame({
        'full_text': g.line.agg('\n\n'.join),
        'n_components': g.size(),
        'n_words': g.words.sum(),
        'n_questions': comp[comp.component_type.eq(3)].groupby('transcriptid').size(),
        'n_analysts': comp[comp.speakertypeid.eq(3)].groupby('transcriptid').speaker.nunique(),
    })
    for sec in ['presentation', 'qa']:
        s = comp[comp.section.eq(sec)].groupby('transcriptid')
        out[f'{sec}_text'] = s.line.agg('\n\n'.join)
        out[f'n_words_{sec}'] = s.words.sum()
    num = ['n_questions', 'n_analysts', 'n_words_presentation', 'n_words_qa']
    out[num] = out[num].fillna(0).astype(int)
    return out.reset_index()


def pull_month(cfg, month, target, batch=500):
    """One calendar month of earnings calls: chosen version's metadata + assembled text."""
    start = pd.Period(month, 'M').start_time
    end = start + pd.offsets.MonthBegin(1)
    country = "and g.isocountry2 = 'US'" if cfg['universe'] == 'us' else ''
    db = connect(cfg)
    try:
        meta = db.raw_sql(META_SQL.format(country=country, order=ORDER[cfg['transcripts']['version_rule']]),
                          params={'event': EARNINGS_CALL, 'start': start.date(), 'end': end.date()},
                          date_cols=['call_date', 'created_utc', 'first_created_utc'])
        ids = meta.transcriptid.astype(int).tolist()
        parts = [db.raw_sql(TEXT_SQL, params={'ids': ids[i:i + batch]}) for i in range(0, len(ids), batch)]
    finally:
        db.close()
    text = (assemble(pd.concat(parts, ignore_index=True)) if parts else
            pd.DataFrame(columns=['transcriptid', 'full_text', 'presentation_text', 'qa_text', 'n_words']))
    meta = meta.drop(columns='transcriptpresentationtypeid').merge(text, on='transcriptid', how='left')
    meta.to_parquet(target, index=False)
    print(f'{month}: {len(meta):,} earnings calls, {meta.full_text.notna().sum():,} with text -> {target}')


def clean(cfg, chunks, calls_target, text_target):
    """Combine monthly pulls, keep the configured period, drop unusable or duplicate calls."""
    df = pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)
    p, t = cfg['period'], cfg['transcripts']
    log = {'pulled': len(df)}
    df = df[df.call_date.between(p['start'], p['end'])]
    log['in period'] = len(df)

    df['call_datetime_utc'] = pd.to_datetime(df.call_date.dt.strftime('%Y-%m-%d') + ' ' + df.call_time_utc.fillna('00:00:00'))
    df['transcript_lag_days'] = (df.created_utc - df.call_datetime_utc).dt.total_seconds() / 86400
    if t['max_lag_days'] is not None:
        df = df[df.transcript_lag_days <= t['max_lag_days']]
        log[f"version created within {t['max_lag_days']} days"] = len(df)
    df = df[df.n_words.fillna(0) >= t['min_words']]
    log[f"at least {t['min_words']} words"] = len(df)

    # CIQ sometimes carries the same call under two keydevids (same company, same date): keep the longer text
    df = df.sort_values(['companyid', 'call_date', 'n_words', 'created_utc'], ascending=[True, True, False, True])
    df = df.drop_duplicates(['companyid', 'call_date'])
    log['after dropping duplicate events'] = len(df)

    text_cols = ['full_text', 'presentation_text', 'qa_text']
    df[['keydevid', 'transcriptid'] + text_cols].to_parquet(text_target, index=False)
    df.drop(columns=text_cols).sort_values(['call_date', 'keydevid']).to_parquet(calls_target, index=False)
    print('Transcripts: ' + ' -> '.join(f'{k} {v:,}' for k, v in log.items()))
