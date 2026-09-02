import pickle, gc, sys, os, time, json
sys.path.insert(0, os.getcwd())
out = {}
for d in ['s16_2_revision_extension_cap','s16_7_name_risk_cap','s16_8_s0recert','s16_8a_opcf_level']:
    t0=time.time()
    r = pickle.load(open(f'outputs/{d}/backtest_result.pkl','rb'))
    rec = {k: getattr(r,k,None) for k in ['optimizer_failure_rate','optimizer_failures','optimizer_fallback_reason_counts','optimizer_rebalances','optimizer_solver_counts','optimizer_solver_fallback_rate','optimizer_solver_fallbacks','optimizer_solver_solves']}
    rec['n_portfolio_weights']=len(r.portfolio_weights)
    out[d]=rec
    print(d, json.dumps(rec, default=str), 'load %.1fs'%(time.time()-t0))
    del r; gc.collect()
json.dump(out, open(os.path.join(os.environ['V'],'W1_T02_fallback.json'),'w'), indent=1, default=str)
