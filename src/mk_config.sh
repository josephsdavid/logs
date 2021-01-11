#!/usr/bin/env bash

# store the project author
author=$1
# whether or not to use git (public or secret, defaults to false)
project_name=$2


print_usage(){
	declare -a help=(
	"you messed up"	)

	printf "%s\n" "${help[@]}" # this is horrible
	exit

}

while getopts a opt ; do
	case "${opt}" in
		a) author=${OPTARG} ;;
		*) print_usage ;;
	esac
done

mkdir -p "$project_name"

echo "author=$author" > "$project_name"/.mdslidesrc

