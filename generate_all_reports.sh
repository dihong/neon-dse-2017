# clean output dir
rm -rf result compiled
mkdir compiled

# Baseline (3/3 tasks were successfully run)
team=Baseline
mkdir compiled/${team}
cd report
python generate_latex.py ${team}
cd ..
cp team_info/${team}/* compiled/${team} 2>/dev/null
cp result/${team}/dse_report.pdf compiled/${team} 2>/dev/null

# BRG (t3 only)
team=BRG
mkdir compiled/${team}
cd report-t3-only
python generate_latex.py ${team}
cd ..
cp team_info/${team}/* compiled/${team} 2>/dev/null
cp result/${team}/dse_report.pdf compiled/${team} 2>/dev/null

# Conor
team=Conor
mkdir compiled/${team}
cd report
python generate_latex.py ${team}
cd ..
cp team_info/${team}/* compiled/${team} 2>/dev/null
cp result/${team}/dse_report.pdf compiled/${team} 2>/dev/null

# FEM (3/3 tasks were successfully run)
team=FEM
mkdir compiled/${team}
cd report
python generate_latex.py ${team}
cd ..
cp team_info/${team}/* compiled/${team} 2>/dev/null
cp result/${team}/dse_report.pdf compiled/${team} 2>/dev/null

# GatorSense (t3 only)
team=GatorSense
mkdir compiled/${team}
cd report-t3-only
python generate_latex.py ${team}
cd ..
cp team_info/${team}/* compiled/${team} 2>/dev/null
cp result/${team}/dse_report.pdf compiled/${team} 2>/dev/null

# Shawn (t1 only)
team=Shawn
mkdir compiled/${team}
cd report-t1-only
python generate_latex.py ${team}
cd ..
cp team_info/${team}/* compiled/${team} 2>/dev/null
cp result/${team}/dse_report.pdf compiled/${team} 2>/dev/null

# StanfordCCB (t3 only)
team=StanfordCCB
mkdir compiled/${team}
cd report-t3-only
python generate_latex.py ${team}
cd ..
cp team_info/${team}/* compiled/${team} 2>/dev/null
cp result/${team}/dse_report.pdf compiled/${team} 2>/dev/null


