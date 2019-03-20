import numpy as np
import cv2
from sklearn.mixture import BayesianGaussianMixture
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

def convertImgIntoRG(filename="GMMSegmentTestImage.jpg", show=False):
    img = cv2.imread(filename)
    rows, cols, dims = img.shape
    B = np.mat(img[:, :, 0])
    G = np.mat(img[:, :, 1])
    R = np.mat(img[:, :, 2])

    # convert R,G,B to r,g
    r = np.array(np.zeros((rows, cols)), dtype='float64')
    g = np.array(np.zeros((rows, cols)), dtype='float64')

    for i in range(rows):
        for j in range(cols):
            Sum = int(R[i, j]) + int(B[i, j]) + int(G[i,j])
            if Sum == 0:
                r[i, j] = 1
                g[i, j] = 1
                break
            else:
                r[i, j] = R[i, j] * 1.0 / (Sum)
                g[i, j] = G[i, j] * 1.0 / (Sum)

    # convert R,G,B to r,g
    # RuntimeWaning: divide by zero encountered in true_divide
    # r = np.true_divide(R ,( R + B + G))
    # g = np.true_divide(G ,( R + B + G))

    if (show):
        alpha = np.mat(np.zeros((rows, cols)),dtype="float32")
        #.alpha = np.true_divide(255, np.max(r,g,1-(r+g)))
        # map back into R,G,B
        for i in range(rows):
             for j in range(cols):
                 max_ = max(r[i, j], g[i, j], 1 - r[i, j] - g[i,j])
                 if(max_ == 0):
                     alpha[i,j] = 255.0
                 else:
                    alpha[i, j] = 255.0 / max_;
        R_out = np.array(np.round(np.multiply(r, alpha)),dtype="int8")
        G_out = np.array(np.round(np.multiply(g, alpha)),dtype="int8")
        B_out = np.array(np.round(np.multiply(alpha, ( 1 - r - g))),dtype="int8")


        img_out = np.array(img)
        for i in range(rows):
            for j in range(cols):
                img_out[i,j,0] = B_out[i,j]
                img_out[i,j,1] = G_out[i,j]
                img_out[i,j,2] = R_out[i,j]
        cv2.imshow("new_img", img_out)
        cv2.waitKey()

    return r, g

def GMM(components_num,rg_show=False,pic_show=False):
    r, g = convertImgIntoRG()
    rows = r.shape[0]
    cols = r.shape[1]

    r_seq = np.reshape(r,(1,-1)).tolist()[0]
    g_seq = np.reshape(g,(1,-1)).tolist()[0]
    #print(r_seq)

    data = [[0, 0] for i in range(len(r_seq))]
    for x in range(len(r_seq)):
        data[x][0] = r_seq[x]
        data[x][1] = g_seq[x]

    plt.scatter(r_seq, g_seq, alpha=0.5)
    plt.xlabel("r")
    plt.ylabel("g")


    mixture_model = BayesianGaussianMixture(
        n_components=components_num,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_distribution',
    )
    mixture_model.fit(data,y=None);

    # plot scatter of rg
    plot_results(mixture_model,
                 r_seq,
                 g_seq,
                 "result"
                 )
    if(rg_show):
        plt.show()

    # compute posterior_probility
    posterior_probility = mixture_model.predict_proba(data)
    imgs = []
    for n in range(components_num):
        img_gray = np.array(np.zeros((rows,cols)), dtype='uint8')
        for i in range(rows):
            for j in range(cols):
                img_gray[j][i] = posterior_probility[i + rows * j][n] * 255
        imgs.append(img_gray)
    if(pic_show):
        for n in range(components_num):
            cv2.imshow("gray picture in component-{}".format(n), imgs[n])

        cv2.waitKey()


def plot_ellipses(ax, weights, means, covars, nstd):
    """
    Given a list of mixture component weights, means, and covariances,
    plot ellipses to show the orientation and scale of the Gaussian mixture dispersal.
    """
    for n in range(means.shape[0]):
        # eigenvector
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])

        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])

        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        # have a question, why use np.sqrt(lambda) rather than sigma
        eig_vals = 2 * nstd * np.sqrt(eig_vals)

        ell = mpl.patches.Ellipse(
            means[n], eig_vals[0], eig_vals[1],
            180 + angle,
            edgecolor=None,)
        ell2 = mpl.patches.Ellipse(
            means[n], eig_vals[0], eig_vals[1],
            180 + angle,
            edgecolor='black',
            fill=False,
            linewidth=1,)
        ell.set_clip_box(ax.bbox)
        ell2.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor('#56B4E9')
        ax.add_artist(ell)
        ax.add_artist(ell2)


def plot_results(model, x, y, title, plot_title=False, nstd=3):
    gs = gridspec.GridSpec(3, 1) # 自定义子图位置
    # ax1:画椭圆
    ax1 = plt.subplot(gs[:, 0])#[0:2, 0])
    means = model.means_
    x_mean = []
    y_mean = []
    for dims in range(means.shape[0]):
        x_mean.append(means[dims][0])
        y_mean.append(means[dims][1])

    ax1.set_title(title)
    ax1.scatter(x, y, s=5, marker='o', alpha=0.8)
    ax1.scatter(x_mean,y_mean, s=40, marker='+')
    ax1.set_xticks(())
    ax1.set_yticks(())

    n_components = model.get_params()['n_components']

    plot_ellipses(ax1, model.weights_, model.means_, model.covariances_, nstd)


    # # ax2:画权重
    # ax2 = plt.subplot(gs[2, 0])
    # ax2.get_xaxis().set_tick_params(direction='out')
    # ax2.yaxis.grid(True, alpha=0.7)
    # for k, w in enumerate(model.weights_):
    #     ax2.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
    #             align='center', edgecolor='black')
    #     ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.),
    #              horizontalalignment='center')
    # ax2.set_xlim(-.6, n_components - .4)
    # ax2.set_ylim(0., 1.1)
    # ax2.tick_params(axis='y', which='both', left='off',
    #                 right='off', labelleft='off')
    # ax2.tick_params(axis='x', which='both', top='off')

    if plot_title:
        ax1.set_ylabel('Estimated Mixtures')
        # ax2.set_ylabel('Weight of each component')


if(__name__ == '__main__'):
    # 1.a show the output of the pitcure after convert into rg and back
    convertImgIntoRG(show=True)
    # 1.b.i scatter the rg picture with GMM 3, 4, 5
    GMM(3, rg_show=True)
    GMM(4, rg_show=True)
    GMM(5, rg_show=True)
    # 1.b.ii show the posterior pics of diff components with GMM 3, 4, 5
    GMM(3, pic_show=True)
    GMM(4, pic_show=True)
    GMM(5, pic_show=True)