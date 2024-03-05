case $1 in
  1)
    python analysis_result.py --ds unbalanced --model blip2
    ;;
  2)
    python analysis_result.py --ds balanced_10 --model blip2
    ;;
  3)
    python analysis_result.py --ds unbalanced --model kosmos
    ;;
  4)
    python analysis_result.py --ds balanced_10 --model kosmos
    ;;
  5)
    python analysis_result.py --ds unbalanced --model lavis
    ;;
  6)
    python analysis_result.py --ds balanced_10 --model lavis
    ;;
  7)
    python analysis_result.py --ds unbalanced --model pretrain_opt6.7b
    ;;
  8)
    python analysis_result.py --ds balanced_10 --model pretrain_opt6.7b
    ;;
  9)
    python analysis_result.py --ds unbalanced --model instructBLIP_flant
    ;;
  10)
    python analysis_result.py --ds balanced_10 --model instructBLIP_flant
    ;;
esac