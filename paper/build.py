"""Compile the Sage manuscript with an installed TeX Live or MiKTeX."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE.parent / 'build')
    parser.add_argument('--documents', nargs='+', default=['main', 'manuscript_review', 'title_page'])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for executable in ('pdflatex', 'bibtex'):
        if not shutil.which(executable):
            raise SystemExit(f'{executable} is required; install TeX Live or MiKTeX.')
    env = os.environ.copy()
    env['BIBINPUTS'] = str(HERE) + os.pathsep + env.get('BIBINPUTS', '')
    env['BSTINPUTS'] = str(HERE) + os.pathsep + env.get('BSTINPUTS', '')
    for name in args.documents:
        if name not in {'main', 'manuscript_review', 'title_page'}:
            raise SystemExit(f'Unsupported document: {name}')
        commands = [('latex1', ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                     '-file-line-error', '-output-directory=' + str(out), name + '.tex'], HERE)]
        if name != 'title_page':
            commands.append(('bibtex', ['bibtex', name], out))
        commands.extend((f'latex{n}', ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                         '-file-line-error', '-output-directory=' + str(out), name + '.tex'], HERE)
                        for n in (2, 3))
        for stage, command, cwd in commands:
            run = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            (out / f'{name}.{stage}.txt').write_text(run.stdout, encoding='utf-8')
            if run.returncode:
                print(run.stdout[-8000:])
                raise SystemExit(f'{name}: {stage} failed ({run.returncode}).')
        print(f'Built {out / (name + ".pdf")}')

if __name__ == '__main__':
    main()
