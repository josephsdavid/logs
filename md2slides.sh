#!/usr/bin/env nix-shell
#! nix-shell -i bash -p R rPackages.rmarkdown rPackages.revealjs texlive.combined.scheme-full

project="$1"
source_file="${2:-readme.md}"
day=$(date '+%B %d')
pdf="false"

author=$( grep author "$project"/.mdslidesrc | awk '{split($0, a, "="); print a[2]}' )

if grep -q  "pdf=true" "$project"/.mdslidesrc ; then
	pdf="true"
fi

echo "$pdf"


rmd_file="$project/presentation.Rmd"

print_usage(){
	declare -a help=(
	"md2slides: convert markdown to presentation"
	"	Usage:"
	"	./md2slides.sh PROJECT_DIR [AUTHORS]"
       	"authors are optional if not it will be David Josephs"
	)

	printf '%s\n' "${help[@]}"
}



if [[ $# -eq 0 ]] ; then
	print_usage
	exit 0
fi

declare -a yaml=(
"---"
"title: $project"
"author: $author"
"date: $day"
"output:"
"  revealjs::revealjs_presentation:"
"    theme: simple"
"---"
)

if [ "$pdf" = "true" ]; then
	declare -a yaml=(
	"---"
	"title: $project"
	"author: $author"
	"date: $day"
	"output:"
	"  beamer_presentation:"
	"    theme: 'Goettingen'"
	"    colortheme: 'beaver'"
	"    fonttheme: 'structurebold'"
	"---"
	)
fi

begincol="<style>.container{display: flex;}.col{flex: 1;}<\/style><div class='container'><div class='col'>"
midcol="<\/div><div class='col'>"
endcol="<\/div><\/div>"



printf '%s\n' "${yaml[@]}" > "$rmd_file"
cd "$project" || exit
cat "$source_file">> "presentation.Rmd"
sed "s/cstart/$begincol/g" presentation.Rmd > presentation2.Rmd
sed "s/cmid/$midcol/g" presentation2.Rmd > presentation3.Rmd
sed "s/cend/$endcol/g" presentation3.Rmd > presentation.Rmd
Rscript -e "rmarkdown::render('presentation.Rmd')" || exit
rm presentation.Rmd
rm presentation2.Rmd
rm presentation3.Rmd
cd .. || exit
