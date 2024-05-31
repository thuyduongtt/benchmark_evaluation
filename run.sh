case $1 in
  1)
    python compute_score.py --model blip2 --ds unbalanced
    ;;
  2)
    python compute_score.py --model blip2 --ds balanced_10
    ;;
  3)
    python compute_score.py --model kosmos --ds unbalanced
    ;;
  4)
    python compute_score.py --model kosmos --ds balanced_10
    ;;
  5)
    python compute_score.py --model lavis --ds unbalanced
    ;;
  6)
    python compute_score.py --model lavis --ds balanced_10
    ;;
  7)
    python compute_score.py --model pretrain_opt6.7b --ds unbalanced
    ;;
  8)
    python compute_score.py --model pretrain_opt6.7b --ds balanced_10
    ;;
  9)
    python compute_score.py --model instructBLIP_flant --ds unbalanced
    ;;
  10)
    python compute_score.py --model instructBLIP_flant --ds balanced_10
    ;;
  11)
    python compute_score.py --model mPLUGOwl2 --ds unbalanced
    ;;
  12)
    python compute_score.py --model mPLUGOwl2 --ds VQAv2
    ;;
  13)
    python compute_score.py --model mPLUGOwl2 --ds OKVQA
    ;;
  14)
    python compute_score.py --model instructBLIP_flant --ds VQAv2
    ;;
  15)
    python compute_score.py --model instructBLIP_flant --ds OKVQA
    ;;
esac