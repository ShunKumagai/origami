# origami
"origami" presents codes and results on computing Teichm\"uller curves (or Veech groups) of origamis.  
See arXiv:2006.00905(https://arxiv.org/abs/2006.00905) for deatails.  

# Requirement
Python 3.8.2 
 
# Note
dic.py calculates the list of isomorphism classes of origamis of given degree (line 172).  
--> output: data_d=[degree].txt  
Replace "array" to "np.array" in the txt and rename it data_d=[degree].py.  
orb.py calculates the orbit decomposition of the T-action and S-action on the set of origamis of given degree (referring data_d=[degree].py).  
--> output: result_d=[degree].py  
orb2.py calculates a simple version of the result.  
--> output: result2_d=[degree].py  

# Author
 Shun Kumagai  
 -Research Center for Pure and Applied Mathematics, 
  Graduate School of Information Sciences, 
  Tohoku University, Sendai 980-8579, Japan.  
 -shun.kumagai.p5@dc.tohoku.ac.jp / syun-kuma@jcom.zaq.ne.jp
