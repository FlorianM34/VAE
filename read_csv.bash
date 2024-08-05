if [ -z "$1"]; then
	column -s, -t < "vae_performance_metrics.csv" | less -#2 -N -S
else
	column -s, -t < "$1" | less -#2 -N -S
fi
