#!/usr/bin/env nix-shell
#! nix-shell -i bash -p R rPackages.rmarkdown rPackages.revealjs

project="$1"
author="${2:-David Josephs}"
source_file="${3:-readme.md}"
day=$(date '+%B %d')

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
"title: $project project status"
"author: $author"
"date: $day"
"output: revealjs::revealjs_presentation"
"---"
)


printf '%s\n' "${yaml[@]}" > "$rmd_file"
cd "$project" || exit
cat "$source_file">> "presentation.Rmd"
Rscript -e "rmarkdown::render('presentation.Rmd')" || exit
rm presentation.Rmd
cd .. || exit
