set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
G=/mnt/sdc/ml/google; DATA=$G/expert_neuron_v9_agentic_code.json; O=/mnt/sdc/ml/std12v5/candidates; mkdir -p $O
LIFT=$(cat $G/eog_v4_t106_lift_pins.txt); BROAD=$(cat /srv/ml/scripts/broad_fk.txt)
P12="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85"
mapfile -t CW < <("$PY" /mnt/sdc/harness/v9_classes.py "$DATA"); CLASSES="${CW[0]}"; WEIGHTS="${CW[1]}"
gen(){ # name fk
  "$PY" /srv/ml/scripts/gen_drop_v5_fk.py --data "$DATA" --target 98 --protect-top 16 --alpha 2.0 --score-mode legacy \
    --strategy max --normalize rank --classes $CLASSES --class-weights $WEIGHTS --protect-strategy same --class-protect-floor 0 \
    --v4-floor-data $G/expert_neuron_base_v7.json --v4-floor-top 0 --v4-floor-map /srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json \
    --breadth-bonus 0.5 --outlier-wnorm-thresh 10000.0 --outlier-mode median \
    --baseline-drop-map /srv/ml/scripts/teacher_force_98e_p16_clean.json --force-keep "$2" --output "$O/$1.json" > "$O/$1.log" 2>&1 || { echo "FAIL $1"; tail -3 "$O/$1.log"; }
}
gen repro_v9_lift120 "$LIFT"
gen A_lift120_plus_fk12 "$LIFT,$P12"
gen B_broad30_plus_fk12 "$BROAD,$P12"
"$PY" - "$O" <<'PY'
import json,sys
O=sys.argv[1]
def L(p):
    d=json.load(open(p)); d=d.get('drop_map',d); return {int(k):set(v) for k,v in d.items() if str(k).isdigit()}
S='/srv/ml/repos/omnimergekit/scripts/'
m={'v9-agentic(Sep)':L('/mnt/sdc/ml/google/v9_agentic_drop_map.json'),'repro':L(O+'/repro_v9_lift120.json'),'A lift120+fk12':L(O+'/A_lift120_plus_fk12.json'),'B broad30+fk12':L(O+'/B_broad30_plus_fk12.json'),'v7map fk12 (tested std12)':L(S+'v8coder_fk12_drop_map.json'),'v7map fk16 (v7-coder)':L(S+'v8coder_fk16_drop_map.json')}
ks=list(m)
for i,a in enumerate(ks):
    for b in ks[i+1:]:
        print(f'{a:26} vs {b:26}: {sum(len(m[a][l]-m[b][l]) for l in m[a]):4d} of 900')
P12=[tuple(map(int,x.split(':'))) for x in "0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85".split(',')]
for k in ks: print(k,'drops of the 12 loop pins:',sum(1 for l,e in P12 if e in m[k][l]))
PY
