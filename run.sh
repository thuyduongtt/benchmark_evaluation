case $1 in
  1)
    python compute_score.py --ds unbalanced --model blip2
    ;;
  2)
    python compute_score.py --ds balanced_10 --model blip2
    ;;
  3)
    python compute_score.py --ds unbalanced --model kosmos
    ;;
  4)
    python compute_score.py --ds balanced_10 --model kosmos
    ;;
  5)
    python compute_score.py --ds unbalanced --model lavis
    ;;
  6)
    python compute_score.py --ds balanced_10 --model lavis
    ;;
  7)
    python compute_score.py --ds unbalanced --model pretrain_opt6.7b
    ;;
  8)
    python compute_score.py --ds balanced_10 --model pretrain_opt6.7b
    ;;
  9)
    python compute_score.py --ds unbalanced --model instructBLIP_flant
    ;;
  10)
    python compute_score.py --ds balanced_10 --model instructBLIP_flant
    ;;
  11)
    python compute_score.py --ds unbalanced --model mPLUGOwl2
    ;;
esac